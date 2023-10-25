import torch
import torchvision.models as models
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor
from fusers import unet, sin_fuser

import numpy as np
from scipy import stats
from tqdm import tqdm
import os
import math
import csv
import copy
import json
from typing import Optional, List

from transformers_tres import Transformer
import data_loader
from posencode import PositionEmbeddingSine

# linear combination between tensors of arbitrary shape
class LinearComb(torch.nn.Module):
	def __init__(self, n: int, len: int):
		super().__init__()
		# to multiply whole tensor by the vector of scalars
		# you need to fill the vector with 1s according to the
		# number of tensor's dims
		# example: [n, d] * [n, 1] will multiply each of the n d-dimensionals vectors by one of the n-s scalars
		# example: [n, s, h, w] * [n, 1, 1, 1]
		to_add = [1 for _ in range(len)]
		self.linear = torch.nn.Parameter(torch.randn([n, *to_add]))
	def forward(self, x):
		return (x*self.linear).sum(dim=1)
	
class TabRFuser(torch.nn.Module):
	def __init__(self, features_dim: int, extra_proj: bool, extra_proj_dim = 0):
		super().__init__()
		self.features_dim = features_dim
		self.extra_proj = extra_proj
		self.extra_proj_dim = extra_proj_dim

		self.W_y = torch.nn.Linear(in_features=1, out_features=features_dim)
		self.test_label_features = torch.nn.Parameter(torch.randn([features_dim]))
		if extra_proj:
			self.W_x = torch.nn.Linear(in_features=features_dim, out_features=extra_proj_dim)
			self.T = torch.nn.Sequential(
				torch.nn.Linear(in_features=extra_proj_dim, out_features=extra_proj_dim*2),
				torch.nn.ReLU(),
				torch.nn.Dropout(p=0.2),
				torch.nn.Linear(in_features=extra_proj_dim*2, out_features=features_dim, bias=False),
			)
		else:
			self.T = torch.nn.Sequential(
				torch.nn.Linear(in_features=features_dim, out_features=features_dim*2),
				torch.nn.ReLU(),
				torch.nn.Dropout(p=0.2),
				torch.nn.Linear(in_features=features_dim*2, out_features=features_dim, bias=False),
			)

	def forward(self, x, nn_s, y):
		if self.extra_proj:
			return (self.T(self.W_x(x) - self.W_x(nn_s)) +self.W_y(y)).sum(dim=1) + self.test_label_features + self.T(torch.zeros([x.shape[0],self.extra_proj_dim], device=x.device))
		else:
			return (self.T(x - nn_s) +self.W_y(y)).sum(dim=1) + self.test_label_features + self.T(torch.zeros([x.shape[0], self.features_dim], device=x.device))

class L2pooling(nn.Module):
	def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
		super(L2pooling, self).__init__()
		self.padding = (filter_size - 2 )//2
		self.stride = stride
		self.channels = channels
		a = np.hanning(filter_size)[1:-1]
		g = torch.Tensor(a[:,None]*a[None,:])
		g = g/torch.sum(g)
		self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
	def forward(self, input):
		input = input**2
		out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
		return (out+1e-12).sqrt()
	
class Net(nn.Module):
	def __init__(self,cfg,device):
		super(Net, self).__init__()
		
		self.device = device
		
		self.cfg = cfg
		self.L2pooling_l1 = L2pooling(channels=256)
		self.L2pooling_l2 = L2pooling(channels=512)
		self.L2pooling_l3 = L2pooling(channels=1024)
		self.L2pooling_l4 = L2pooling(channels=2048)

		if cfg.middle_fuse and cfg.double_branch:
			self.L2pooling_l1_2 = L2pooling(channels=256)
			self.L2pooling_l2_2 = L2pooling(channels=512)
			self.L2pooling_l3_2 = L2pooling(channels=1024)
			self.L2pooling_l4_2 = L2pooling(channels=2048)
		
		if cfg.single_channel and not cfg.finetune:
			if cfg.unet:
				self.initial_fuser = unet.IQAUNetModel(
					image_size=(224, 224),
					in_channels= 3*(cfg.k+1),
					model_channels=cfg.model_channels,
					out_channels=3,
					k = cfg.k,
					num_res_blocks=1,
					attention_resolutions= cfg.attention_resolutions,
					scaling_factors=cfg.scaling_factors,
					num_heads=1,
					resblock_updown=False,
					conv_resample=True,
    				first_conv_resample=cfg.first_conv_resample,
					channel_mult=cfg.channel_mult,
					middle_attention=cfg.middle_attention
				)
			elif cfg.sin:
				self.initial_fuser = sin_fuser.SinFuser(
					k = cfg.k,
					before_initial_conv=cfg.before_conv_in_sin
				)
			elif cfg.conv1x1:
				self.initial_fuser = nn.Conv2d(3*(cfg.k+1), 3, kernel_size=(1, 1), bias=cfg.conv_bias)


			
		if cfg.network =='resnet50':
			from resnet_modify  import resnet50 as resnet_modifyresnet
			dim_modelt = 3840
			modelpretrain = models.resnet50(weights="DEFAULT")

			# multichannel input to TReS instead of 3-channeled
			if not cfg.single_channel:
				if cfg.k > 0 and not cfg.finetune:
					modelpretrain.conv1 = nn.Conv2d(3*(cfg.k+1), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		elif cfg.network =='resnet34':
			from resnet_modify  import resnet34 as resnet_modifyresnet
			modelpretrain = models.resnet34(weights="DEFAULT")
			dim_modelt = 960
			self.L2pooling_l1 = L2pooling(channels=64)
			self.L2pooling_l2 = L2pooling(channels=128)
			self.L2pooling_l3 = L2pooling(channels=256)
			self.L2pooling_l4 = L2pooling(channels=512)
		elif cfg.network == 'resnet18':
			from resnet_modify  import resnet18 as resnet_modifyresnet
			modelpretrain = models.resnet18(weights="DEFAULT")
			dim_modelt = 960
			self.L2pooling_l1 = L2pooling(channels=64)
			self.L2pooling_l2 = L2pooling(channels=128)
			self.L2pooling_l3 = L2pooling(channels=256)
			self.L2pooling_l4 = L2pooling(channels=512)


		torch.save(modelpretrain.state_dict(), 'modelpretrain')
		
		if not cfg.single_channel:
			if not cfg.finetune:
				self.model = resnet_modifyresnet(k=cfg.k)
			else:
				self.model = resnet_modifyresnet(k=0)
		else:
			self.model = resnet_modifyresnet(k=0)
			if cfg.middle_fuse and cfg.double_branch:
				self.model_2 = resnet_modifyresnet(k=0)
		self.model.load_state_dict(torch.load('modelpretrain'), strict=True)
		if cfg.middle_fuse and cfg.double_branch:
			self.model_2.load_state_dict(torch.load('modelpretrain'), strict=True)

		self.dim_modelt = dim_modelt

		os.remove("modelpretrain")
		
		nheadt=cfg.nheadt
		num_encoder_layerst=cfg.num_encoder_layerst
		dim_feedforwardt=cfg.dim_feedforwardt
		ddropout=0.5
		normalize =True
			
			
		self.transformer = Transformer(d_model=dim_modelt,nhead=nheadt,
									   num_encoder_layers=num_encoder_layerst,
									   dim_feedforward=dim_feedforwardt,
									   normalize_before=normalize,
									   dropout = ddropout)
		if cfg.middle_fuse and cfg.double_branch:
			self.transformer_2 = Transformer(d_model=dim_modelt,nhead=nheadt,
									   num_encoder_layers=num_encoder_layerst,
									   dim_feedforward=dim_feedforwardt,
									   normalize_before=normalize,
									   dropout = ddropout)

		self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)

		if (cfg.dataset == 'spaq' or cfg.cross_dataset == 'spaq') and cfg.use_metainfo:
			if cfg.metainfo_aggregation == 'cat':
				self.preprocess_meta_info = nn.Sequential(
					nn.Linear(20, 40),
					nn.SiLU(),
					nn.Linear(40, 20)
				)

		if cfg.use_metainfo and cfg.dataset == 'spaq' and cfg.metainfo_aggregation == 'cat':
			self.fc2 = nn.Linear(dim_modelt+20, self.model.fc.in_features+20)
		elif cfg.middle_label_fuse:
			if cfg.middle_label_aggregation == 'cat':
				self.fc2 = nn.Linear(dim_modelt+cfg.middle_label_aggregation_dim, self.model.fc.in_features+cfg.middle_label_aggregation_dim)
			elif cfg.middle_label_aggregation == 'sum':
				self.fc2 = nn.Linear(dim_modelt, self.model.fc.in_features)
			else:
				self.fc2 = nn.Linear(dim_modelt+1, self.model.fc.in_features+1)
		else:
			self.fc2 = nn.Linear(dim_modelt, self.model.fc.in_features) 

		if not cfg.single_channel:
			if not cfg.finetune:
				if cfg.multi_return:
					self.fc = nn.Linear(self.model.fc.in_features*2, cfg.k+1)
				else:
					self.fc = nn.Linear(self.model.fc.in_features*2, 1)
			else:
				self.fc = nn.Linear(self.model.fc.in_features*2, 1)
		else:
			if cfg.use_metainfo and cfg.dataset == 'spaq' and cfg.metainfo_aggregation == 'cat':
				self.fc = nn.Linear((self.model.fc.in_features+20)*2, 1)
			elif cfg.middle_label_fuse:
				if cfg.middle_label_aggregation == 'cat':
					self.fc = nn.Linear((self.model.fc.in_features + cfg.middle_label_aggregation_dim)*2, 1)
				elif cfg.middle_label_aggregation == 'sum':
					self.fc = nn.Linear(self.model.fc.in_features*2, 1)
				else:
					self.fc = nn.Linear((self.model.fc.in_features+1)*2, 1)
			else:
				self.fc = nn.Linear(self.model.fc.in_features*2, 1)

		# 16/10/2023 version of late fuse
		if cfg.late_fuse:
			self.final_fuser = nn.Sequential(
				nn.Linear(2, 8),
				nn.SiLU(),
				nn.Linear(8, 1)
			)

		if cfg.middle_label_fuse:
			if cfg.middle_label_aggregation == 'cat':
				self.middle_label_embedder = nn.Sequential(
					nn.Linear(1, cfg.middle_label_aggregation_dim//2),
					nn.SiLU(),
					nn.Linear(cfg.middle_label_aggregation_dim//2, cfg.middle_label_aggregation_dim)
				)
			elif cfg.middle_label_aggregation == 'sum':
				self.first_middle_label_embedder = nn.Sequential(
					nn.Linear(1, 10),
					nn.SiLU(),
					nn.Linear(10, dim_modelt)
				)
				self.second_middle_label_embedder = nn.Sequential(
					nn.Linear(1, 10),
					nn.SiLU(),
					nn.Linear(10, self.model.fc.in_features)
				)

		if cfg.weight_before_late_fuse:
			self.weighter = nn.Sequential(
				nn.Linear(cfg.k_late, cfg.k_late*2),
				nn.SiLU(),
				nn.Linear(cfg.k_late*2, 1)
			)
		
		if cfg.middle_fuse:
			if cfg.attention_in_middle_fuse:
				self.first_middle_fuser = torch.nn.MultiheadAttention(embed_dim=dim_modelt, num_heads=4, kdim=dim_modelt, vdim=dim_modelt, batch_first=True)
				self.second_middle_fuser = torch.nn.MultiheadAttention(embed_dim=self.model.fc.in_features, num_heads=4, kdim=self.model.fc.in_features, vdim=self.model.fc.in_features, batch_first=True)
			else:	
				self.first_middle_fuser = LinearComb(cfg.k + 1, 1)
				self.second_middle_fuser = LinearComb(cfg.k + 1, 1)
			self.consist1_fuser = [
				LinearComb(cfg.k + 1, 3).to(device),
				LinearComb(cfg.k + 1, 3).to(device)
			]
			self.consist2_fuser = [
				LinearComb(cfg.k + 1, 3).to(device),
				LinearComb(cfg.k + 1, 3).to(device)
			]
		elif cfg.tabr_fuse:
			self.first_tabr_fuser = TabRFuser(
				features_dim=dim_modelt, 
				extra_proj=cfg.tabr_extra_proj, extra_proj_dim=cfg.tabr_extra_proj_dim
			)
			self.second_tabr_fuser = TabRFuser(
				features_dim=self.model.fc.in_features, 
				extra_proj=cfg.tabr_extra_proj, extra_proj_dim=cfg.tabr_extra_proj_dim
			)
		

		self.avg7 = nn.AvgPool2d((7, 7))
		self.avg8 = nn.AvgPool2d((8, 8))
		self.avg4 = nn.AvgPool2d((4, 4))
		self.avg2 = nn.AvgPool2d((2, 2))		   
		
		self.drop2d = nn.Dropout(p=0.1)

		if cfg.middle_fuse and cfg.double_branch:
			self.avg7_2 = nn.AvgPool2d((7, 7))
			self.avg8_2 = nn.AvgPool2d((8, 8))
			self.avg4_2 = nn.AvgPool2d((4, 4))
			self.avg2_2 = nn.AvgPool2d((2, 2))		   
			
			self.drop2d_2 = nn.Dropout(p=0.1)

		self.consistency = nn.L1Loss()
		

	def forward(self, x, t=0, info=[]):
		self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7).to(self.device))
		if self.cfg.middle_fuse:
			if self.cfg.double_branch:
				self.pos_enc = self.pos_enc_1.repeat(x.shape[0],1,1,1).contiguous()
				self.pos_enc_2 = self.pos_enc_1.repeat(x.shape[0]* (self.cfg.k),1,1,1).contiguous()
			else:
				self.pos_enc = self.pos_enc_1.repeat(x.shape[0]* (self.cfg.k + 1),1,1,1).contiguous()
		elif self.cfg.tabr_fuse:
			self.pos_enc = self.pos_enc_1.repeat(x.shape[0]* (self.cfg.k + 1),1,1,1).contiguous()
			self.pos_enc_2 = self.pos_enc_1.repeat(x.shape[0],1,1,1).contiguous()
		else:
			self.pos_enc = self.pos_enc_1.repeat(x.shape[0],1,1,1).contiguous()
		batch_size = x.shape[0]

		# preproccess spaq's metainfo
		if (self.cfg.dataset == 'spaq' or self.cfg.cross_dataset == 'spaq') and self.cfg.use_metainfo:
			preprocessed_meta_info = self.preprocess_meta_info(
				info.reshape([batch_size * self.cfg.k, -1])
			).reshape([batch_size, self.cfg.k, -1])
			# pad with zeroes
			preprocessed_meta_info = torch.cat([torch.zeros([batch_size, 1 ,preprocessed_meta_info.shape[-1]], device=self.device), preprocessed_meta_info], dim=1)

		# preproccess labels if middle label fuse:
		if self.cfg.middle_label_fuse:
			# if we concat mlp-ed labels
			if self.cfg.middle_label_aggregation == 'cat':
				preprocessed_labels = self.middle_label_embedder(
					t[:, :self.cfg.k].reshape([batch_size * self.cfg.k, 1])
				).reshape([batch_size, self.cfg.k, self.cfg.middle_label_aggregation_dim])
				preprocessed_labels = torch.cat([torch.zeros([batch_size, 1 ,preprocessed_labels.shape[-1]], device=self.device), preprocessed_labels], dim=1)
			# if we sum mlp-ed labels
			elif self.cfg.middle_label_aggregation == 'sum':
				preprocessed_labels_1 = self.first_middle_label_embedder(
					t[:, :self.cfg.k].reshape([batch_size * self.cfg.k, 1])
				).reshape([batch_size, self.cfg.k, self.dim_modelt])
				preprocessed_labels_2 = self.second_middle_label_embedder(
					t[:, :self.cfg.k].reshape([batch_size * self.cfg.k, 1])
				).reshape([batch_size, self.cfg.k, self.model.fc.in_features])
				preprocessed_labels_1 = torch.cat([torch.zeros([batch_size, 1 ,preprocessed_labels_1.shape[-1]], device=self.device), preprocessed_labels_1], dim=1)
				preprocessed_labels_2 = torch.cat([torch.zeros([batch_size, 1 ,preprocessed_labels_2.shape[-1]], device=self.device), preprocessed_labels_2], dim=1)
			# if we concat labels
			else:
				preprocessed_labels = torch.cat([-1*torch.ones([batch_size, 1, 1 ], device=self.device), t[:, :self.cfg.k].unsqueeze(-1)], dim=1)
				

		if self.cfg.single_channel:
			if self.cfg.unet or self.cfg.sin:
				# unet and sin fusers eat [b, 3*(k+1), h, w] shaped tensors with labels
				# and outputs [b, 3, h, w]
				# here k must be equal to k_late
				x = self.initial_fuser(x,t)

			elif self.cfg.conv1x1:
				# 1x1 conv fuser. Takes [b, 3*(k+1), h, w] with no labels, outputs [b, 3, h, w]
				x = self.initial_fuser(x)

			elif self.cfg.middle_fuse:
				if self.cfg.double_branch:
					# double branch middle fuse needs two input separate tensors:
					# 1) with original pic 2) with k neighbours pics
					# [b, 3*(k+1), h, w] -> [b, 3, h, w] , [b*k, 3, h, w]
					x_1 = x[::, :3, ::, :: ] #  -> [b, 3, h, w]
					x_2 = x[::, 3:, ::, :: ].reshape([batch_size * (self.cfg.k), 3, self.cfg.patch_size, self.cfg.patch_size]) # -> [b*k, 3, h, w]
				else:
					# single branch middle fuse proccess original pic and its neighbours
					# in parallel, which is obtained using big batch
					# [b, 3*(k+1), h, w] -> [b*(k+1), 3, h, w] 
					x = x.reshape([batch_size * (self.cfg.k + 1), 3, self.cfg.patch_size, self.cfg.patch_size])

			# tabr-inspired fuser. needs parallel processing of
			# original pic and its neighbours
			elif self.cfg.tabr_fuse:
				x = x.reshape([batch_size * (self.cfg.k + 1), 3, self.cfg.patch_size, self.cfg.patch_size])

			# vanilla TReS
			# take only original pic
			else:
				x = x[::, :3, ::, :: ]

		# double branch handler
		if self.cfg.double_branch:
			# main branch
			_,layer1,layer2,layer3,layer4_1 = self.model(x_1) 

			layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1,dim=1, p=2))))
			layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2,dim=1, p=2))))
			layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3,dim=1, p=2))))
			layer4_t =           self.drop2d(self.L2pooling_l4(F.normalize(layer4_1,dim=1, p=2)))
			layers = torch.cat((layer1_t,layer2_t,layer3_t,layer4_t),dim=1)
			out_t_c_1 = self.transformer(layers,self.pos_enc)
			out_t_o_1 = torch.flatten(self.avg7(out_t_c_1),start_dim=1)
			layer4_o = self.avg7(layer4_1)
			layer4_o_1 = torch.flatten(layer4_o,start_dim=1)

			# neighbours branch
			_,layer1,layer2,layer3,layer4_2 = self.model_2(x_2) 

			layer1_t = self.avg8_2(self.drop2d_2(self.L2pooling_l1_2(F.normalize(layer1,dim=1, p=2))))
			layer2_t = self.avg4_2(self.drop2d_2(self.L2pooling_l2_2(F.normalize(layer2,dim=1, p=2))))
			layer3_t = self.avg2_2(self.drop2d_2(self.L2pooling_l3_2(F.normalize(layer3,dim=1, p=2))))
			layer4_t =           self.drop2d_2(self.L2pooling_l4_2(F.normalize(layer4_2,dim=1, p=2)))
			layers = torch.cat((layer1_t,layer2_t,layer3_t,layer4_t),dim=1)
			out_t_c_2 = self.transformer_2(layers,self.pos_enc_2)
			out_t_o_2 = torch.flatten(self.avg7_2(out_t_c_2),start_dim=1)
			layer4_o = self.avg7_2(layer4_2)
			layer4_o_2 = torch.flatten(layer4_o,start_dim=1)

			# concat branches to [b*(k+1), 3, h, w] shape
			out_t_o = torch.cat([out_t_o_1, out_t_o_2], dim=0)
			out_t_c = torch.cat([out_t_c_1, out_t_c_2], dim=0)
			layer4_o = torch.cat([layer4_o_1, layer4_o_2], dim=0)
			layer4 = torch.cat([layer4_1, layer4_2], dim=0)

		# standard single branch
		else:
			_,layer1,layer2,layer3,layer4 = self.model(x) 

			layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1,dim=1, p=2))))
			layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2,dim=1, p=2))))
			layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3,dim=1, p=2))))
			layer4_t =           self.drop2d(self.L2pooling_l4(F.normalize(layer4,dim=1, p=2)))
			layers = torch.cat((layer1_t,layer2_t,layer3_t,layer4_t),dim=1)


			out_t_c = self.transformer(layers,self.pos_enc)
			out_t_o = torch.flatten(self.avg7(out_t_c),start_dim=1)

			layer4_o = self.avg7(layer4)
			layer4_o = torch.flatten(layer4_o,start_dim=1)

		# first middle fuse
		# fuse out_t_o before fc : 
		# 1) = reshape: [b*(k+1), d] -> [b, (k+1), d]
		# 2) = weighted sum: [b, (k+1), d] -> [b, d]
		if self.cfg.middle_fuse:
			out_t_o = out_t_o.reshape([batch_size, self.cfg.k + 1, -1])
			if self.cfg.attention_in_middle_fuse:
				out_t_o = self.first_middle_fuser(out_t_o[::, :1, ::], out_t_o[::, 1:, ::],  out_t_o[::, 1:, ::])[0]
			else:
				# spaq's metainfo fuse
				if self.cfg.dataset == 'spaq':
					if self.cfg.metainfo_aggregation == 'sum':
						out_t_o += preprocessed_meta_info
					elif self.cfg.metainfo_aggregation == 'cat':
						out_t_o = torch.cat([out_t_o, preprocessed_meta_info], dim=-1)
			
				# middle label fuse
				if self.cfg.middle_label_fuse:
					if self.cfg.middle_label_aggregation == 'cat':
						out_t_o = torch.cat([out_t_o, preprocessed_labels], dim=-1)
					elif self.cfg.middle_label_aggregation == 'sum':
						out_t_o += preprocessed_labels_1
					else:
						out_t_o = torch.cat([out_t_o, preprocessed_labels], dim=-1)
				out_t_o = self.first_middle_fuser(out_t_o)
		elif self.cfg.tabr_fuse:
			out_t_o = out_t_o.reshape([batch_size, self.cfg.k + 1, -1])
			out_t_o = self.first_tabr_fuser(
				x=out_t_o[:, :1, :],
				nn_s=out_t_o[:, 1:, :],
				y=t[:, :self.cfg.k, None]
			)
				
		out_t_o = self.fc2(out_t_o)

		# second middle fuse
		# fuse layer4_o before concat with out_t_o and fc:
		# 1) = reshape: [b*(k+1), d] -> [b, (k+1), d]
		# 2) = weighted sum: [b, (k+1), d] -> [b, d]
		if self.cfg.middle_fuse:
			layer4_o = layer4_o.reshape([batch_size, self.cfg.k + 1, -1])
			if self.cfg.attention_in_middle_fuse:
				layer4_o = self.second_middle_fuser(layer4_o[::, :1, ::], layer4_o[::, 1:, ::],  layer4_o[::, 1:, ::])[0]
			else:
				if self.cfg.dataset == 'spaq':
					if self.cfg.metainfo_aggregation == 'sum':
						layer4_o += preprocessed_meta_info
					elif self.cfg.metainfo_aggregation == 'cat':
						layer4_o = torch.cat([layer4_o, preprocessed_meta_info], dim=-1)

				# middle label fuse
				if self.cfg.middle_label_fuse:
					if self.cfg.middle_label_aggregation == 'cat':
						layer4_o = torch.cat([layer4_o, preprocessed_labels], dim=-1)
					elif self.cfg.middle_label_aggregation == 'sum':
						layer4_o += preprocessed_labels_2
					else:
						layer4_o = torch.cat([layer4_o, preprocessed_labels], dim=-1)
				layer4_o = self.second_middle_fuser(layer4_o)
		elif self.cfg.tabr_fuse:
			layer4_o = layer4_o.reshape([batch_size, self.cfg.k + 1, -1])
			layer4_o = self.second_tabr_fuser(
				x=layer4_o[:, :1, :],
				nn_s=layer4_o[:, 1:, :],
				y=t[:, :self.cfg.k, None]
			)

		# backbone output
		predictionQA = self.fc(torch.flatten(torch.cat((out_t_o,layer4_o),dim=1),start_dim=1))

		# fuse backbone's output with neighbours' labels
		if self.cfg.late_fuse:
			if self.cfg.weight_before_late_fuse:
				t = self.weighter(t)
			else:
				t = t.mean(dim=1).unsqueeze(1)
			labels = torch.cat([predictionQA, t], dim=1)
			predictionQA = self.final_fuser(labels)

		# =============================================================================
		# =============================================================================

		# double branch handler
		if self.cfg.double_branch:

			# main branch
			_,flayer1,flayer2,flayer3,flayer4_1 = self.model(torch.flip(x_1, [3])) 
			flayer1_t = self.avg8( self.L2pooling_l1(F.normalize(flayer1,dim=1, p=2)))
			flayer2_t = self.avg4( self.L2pooling_l2(F.normalize(flayer2,dim=1, p=2)))
			flayer3_t = self.avg2( self.L2pooling_l3(F.normalize(flayer3,dim=1, p=2)))
			flayer4_t =            self.L2pooling_l4(F.normalize(flayer4_1,dim=1, p=2))
			flayers = torch.cat((flayer1_t,flayer2_t,flayer3_t,flayer4_t),dim=1)
			fout_t_c_1 = self.transformer(flayers,self.pos_enc)

			# neighbours' branch
			_,flayer1,flayer2,flayer3,flayer4_2 = self.model_2(torch.flip(x_2, [3])) 
			flayer1_t = self.avg8_2( self.L2pooling_l1_2(F.normalize(flayer1,dim=1, p=2)))
			flayer2_t = self.avg4_2( self.L2pooling_l2_2(F.normalize(flayer2,dim=1, p=2)))
			flayer3_t = self.avg2_2( self.L2pooling_l3_2(F.normalize(flayer3,dim=1, p=2)))
			flayer4_t =            self.L2pooling_l4_2(F.normalize(flayer4_2,dim=1, p=2))
			flayers = torch.cat((flayer1_t,flayer2_t,flayer3_t,flayer4_t),dim=1)
			fout_t_c_2 = self.transformer(flayers,self.pos_enc_2)

			# concat branches
			fout_t_c = torch.cat([fout_t_c_1, fout_t_c_2], dim=0)
			flayer4 = torch.cat([flayer4_1, flayer4_2], dim=0)


		else:
			_,flayer1,flayer2,flayer3,flayer4 = self.model(torch.flip(x, [3])) 
			flayer1_t = self.avg8( self.L2pooling_l1(F.normalize(flayer1,dim=1, p=2)))
			flayer2_t = self.avg4( self.L2pooling_l2(F.normalize(flayer2,dim=1, p=2)))
			flayer3_t = self.avg2( self.L2pooling_l3(F.normalize(flayer3,dim=1, p=2)))
			flayer4_t =            self.L2pooling_l4(F.normalize(flayer4,dim=1, p=2))
			flayers = torch.cat((flayer1_t,flayer2_t,flayer3_t,flayer4_t),dim=1)
			fout_t_c = self.transformer(flayers,self.pos_enc)
		

		if self.cfg.middle_fuse:
			out_t_c = self.consist1_fuser[0](out_t_c.reshape([batch_size, self.cfg.k + 1, *out_t_c.shape[1:]]))
			fout_t_c = self.consist1_fuser[1](fout_t_c.reshape([batch_size, self.cfg.k + 1, *fout_t_c.shape[1:]])).detach()
			layer4 = self.consist2_fuser[0](layer4.reshape([batch_size, self.cfg.k + 1, *layer4.shape[1:]]))
			flayer4 = self.consist2_fuser[1](flayer4.reshape([batch_size, self.cfg.k + 1, *flayer4.shape[1:]])).detach()

		consistloss1 = self.consistency(out_t_c,fout_t_c)
		consistloss2 = self.consistency(layer4,flayer4)
		consistloss = 1*(consistloss1+consistloss2)
				
		return predictionQA, consistloss


class TReS(object):
	
	def __init__(self, config, device,  svPath, datapath, train_idx, test_idx,Net):
		super(TReS, self).__init__()
		
		self.device = device
		self.epochs = config.epochs
		self.test_patch_num = config.test_patch_num
		self.l1_loss = torch.nn.L1Loss()
		self.lr = config.lr
		self.lrratio = config.lrratio
		self.weight_decay = config.weight_decay
		self.net = Net(config,device).to(device) 

		# load checkpoint, change architecture and freeze internal params
		if not config.single_channel:
			if config.finetune:
				# load checkpoint,
				self.net.load_state_dict(torch.load(config.ckpt, map_location=device))
				# change architecture
				self.net.model.conv1 = nn.Conv2d(3*(config.k+1), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
				if config.multi_return:
					self.net.fc = nn.Linear(self.net.model.fc.in_features*2, config.k+1)
				# freeze internal params
				if not config.full_finetune:
					for parameter in self.net.parameters():
						parameter.requires_grad = False
					for parameter in self.net.model.conv1.parameters():
						parameter.requires_grad = True
					if config.multi_return:
						for parameter in self.net.fc.parameters():
							parameter.requires_grad = True
				self.net.to(device)
		else:
			if config.finetune:
				self.net.load_state_dict(torch.load(config.ckpt, map_location=device))
				self.net.initial_fuser = nn.Conv2d(3*(config.k+1), 3, 1, bias=config.conv_bias)
				if not config.full_finetune:
					for parameter in self.net.parameters():
						parameter.requires_grad = False
					for parameter in self.net.initial_fuser.parameters():
						parameter.requires_grad = True
				self.net.to(device)

		
		# resume only with constant lr scheduler
		if config.resume:
			self.net.load_state_dict(torch.load(config.ckpt, map_location=device))
				

		self.droplr = config.droplr
		self.config = config
		self.paras = [{'params': self.net.parameters(), 'lr': self.lr} ]
		if config.optimizer == "adam":
			self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
		elif config.optimizer == "radam":
			self.solver = torch.optim.RAdam(self.paras, weight_decay=self.weight_decay)
		elif config.optimizer == "sgd":
			self.solver = torch.optim.SGD(self.paras, weight_decay=self.weight_decay)

		if config.scheduler == "log":
			self.scheduler = torch.optim.lr_scheduler.StepLR(self.solver, step_size=self.droplr, gamma=self.lrratio)
		if config.scheduler == "cosine":
			self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.solver, T_max=config.T_max, eta_min=config.eta_min)


		train_loader = data_loader.DataLoader(config.dataset, datapath, 
											  train_idx, config.patch_size, 
											  config.train_patch_num,
											  seed=config.seed, k=config.k, 
											  batch_size=config.batch_size, istrain=True,
											  cross_root=config.cross_datapath, cross_dataset=config.cross_dataset,
											  delimeter=config.delimeter, retrieve_size=config.retrieve_size)
		
		test_loader = data_loader.DataLoader(config.dataset, datapath,
											 test_idx, config.patch_size,
											 config.test_patch_num,
											 seed=config.seed, k=config.k, istrain=False,
											 cross_root=config.cross_datapath, cross_dataset=config.cross_dataset,
											 delimeter=config.delimeter, retrieve_size=config.retrieve_size)
		
		self.train_data = train_loader.get_data()
		self.test_data = test_loader.get_data()


		
		
	def train(self,seed,svPath):
		best_srcc = 0.0
		best_plcc = 0.0
		print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tLearning_Rate\tdroplr')
		steps = 0
		results = {}
		train_results ={}
		performPath = svPath +'/' + 'val_SRCC_PLCC_'+str(self.config.vesion)+'_'+str(seed)+'.json'
		trainPerformPath = svPath +'/' + 'train_LOSS_SRCC_'+str(self.config.vesion)+'_'+str(seed)+'.json'
		with open(performPath, 'w') as json_file2:
			json.dump(  {} , json_file2)
		with open(trainPerformPath, 'w') as json_file3:
			json.dump( {}, json_file3 )
		
		for epochnum in range(self.epochs):
			self.net.train()
			epoch_loss = []
			pred_scores = []
			gt_scores = []
			pbar = tqdm(self.train_data, leave=False)

			for i, (img, label, info) in enumerate(pbar):
				img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
				label = torch.as_tensor(label.to(self.device)).requires_grad_(False)
				info = torch.as_tensor(info.to(self.device)).requires_grad_(False)

				steps+=1
				
				self.net.zero_grad()
			
				# if we use fuser that processes labels
				# labels must be transferred as the input as well
				# ! MUST NOT USE 0-th label because it's orginal label !
				# ! DON'T CONFUSE WITH NOT TAKING 0-th label at the preprocess stage ! 
				pred,closs = self.net(img, label[::, 1:self.config.k_late+1], info)
				pred2,closs2 = self.net(torch.flip(img, [3]), label[::, 1:self.config.k_late+1], info)
				 
				if self.config.multi_return:
					pred_scores = pred_scores + pred[:,0].flatten().cpu().tolist()
					gt_scores = gt_scores + label[:,0].flatten().cpu().tolist()
					loss_qa = self.l1_loss(pred, label.float().detach())
					loss_qa2 = self.l1_loss(pred2, label.float().detach())

					# =============================================================================
					# =============================================================================

					if not self.config.multi_ranking:

						indexlabel = torch.argsort(label, dim=0)[:, 0].flatten() # small--> large
						anchor1 = torch.unsqueeze(pred.T[0, indexlabel[0],...].contiguous(),dim=0) # d_min
						positive1 = torch.unsqueeze(pred.T[0, indexlabel[1],...].contiguous(),dim=0) # d'_min+
						negative1_1 = torch.unsqueeze(pred.T[0, indexlabel[-1],...].contiguous(),dim=0) # d_max+

						
						anchor2 = torch.unsqueeze(pred.T[0, indexlabel[-1],...].contiguous(),dim=0)# d_max
						positive2 = torch.unsqueeze(pred.T[0, indexlabel[-2],...].contiguous(),dim=0)# d'_max+
						negative2_1 = torch.unsqueeze(pred.T[0, indexlabel[0],...].contiguous(),dim=0)# d_min+

						# =============================================================================
						# =============================================================================

						fanchor1 = torch.unsqueeze(pred2.T[0, indexlabel[0],...].contiguous(),dim=0)
						fpositive1 = torch.unsqueeze(pred2.T[0, indexlabel[1],...].contiguous(),dim=0)
						fnegative1_1 = torch.unsqueeze(pred2.T[0,indexlabel[-1],...].contiguous(),dim=0)

						fanchor2 = torch.unsqueeze(pred2.T[0, indexlabel[-1],...].contiguous(),dim=0)
						fpositive2 = torch.unsqueeze(pred2.T[0, indexlabel[-2],...].contiguous(),dim=0)
						fnegative2_1 = torch.unsqueeze(pred2.T[0, indexlabel[0],...].contiguous(),dim=0)

						assert (label.T[0, indexlabel[-1]]-label.T[0, indexlabel[1]])>=0
						assert (label.T[0, indexlabel[-2]]-label.T[0, indexlabel[0]])>=0
						triplet_loss1 = nn.TripletMarginLoss(margin=(label.T[0, indexlabel[-1]]-label.T[0, indexlabel[1]]), p=1) # d_min,d'_min,d_max
						triplet_loss2 = nn.TripletMarginLoss(margin=(label.T[0, indexlabel[-2]]-label.T[0, indexlabel[0]]), p=1)
						
						tripletlosses = triplet_loss1(anchor1, positive1, negative1_1) + \
						triplet_loss2(anchor2, positive2, negative2_1)
						ftripletlosses = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
						triplet_loss2(fanchor2, fpositive2, fnegative2_1)

					else:
						tripletlosses = torch.zeros([self.config.k])
						ftripletlosses = torch.zeros([self.config.k])

						for l in range(self.config.k):
							indexlabel = torch.argsort(label, dim=0)[:, l].flatten() # small--> large
							anchor1 = torch.unsqueeze(pred.T[l, indexlabel[0],...].contiguous(),dim=0) # d_min
							positive1 = torch.unsqueeze(pred.T[l, indexlabel[1],...].contiguous(),dim=0) # d'_min+
							negative1_1 = torch.unsqueeze(pred.T[l, indexlabel[-1],...].contiguous(),dim=0) # d_max+

							
							anchor2 = torch.unsqueeze(pred.T[l, indexlabel[-1],...].contiguous(),dim=0)# d_max
							positive2 = torch.unsqueeze(pred.T[l, indexlabel[-2],...].contiguous(),dim=0)# d'_max+
							negative2_1 = torch.unsqueeze(pred.T[l, indexlabel[0],...].contiguous(),dim=0)# d_min+

							# =============================================================================
							# =============================================================================

							fanchor1 = torch.unsqueeze(pred2.T[l, indexlabel[0],...].contiguous(),dim=0)
							fpositive1 = torch.unsqueeze(pred2.T[l, indexlabel[1],...].contiguous(),dim=0)
							fnegative1_1 = torch.unsqueeze(pred2.T[l,indexlabel[-1],...].contiguous(),dim=0)

							fanchor2 = torch.unsqueeze(pred2.T[l, indexlabel[-1],...].contiguous(),dim=0)
							fpositive2 = torch.unsqueeze(pred2.T[l, indexlabel[-2],...].contiguous(),dim=0)
							fnegative2_1 = torch.unsqueeze(pred2.T[l, indexlabel[0],...].contiguous(),dim=0)

							assert (label.T[l, indexlabel[-1]]-label.T[l, indexlabel[1]])>=0
							assert (label.T[l, indexlabel[-2]]-label.T[l, indexlabel[0]])>=0
							triplet_loss1 = nn.TripletMarginLoss(margin=(label.T[l, indexlabel[-1]]-label.T[l, indexlabel[1]]), p=1) # d_min,d'_min,d_max
							triplet_loss2 = nn.TripletMarginLoss(margin=(label.T[l, indexlabel[-2]]-label.T[l, indexlabel[0]]), p=1)

							tripletlosses[l] = triplet_loss1(anchor1, positive1, negative1_1) + \
							triplet_loss2(anchor2, positive2, negative2_1)

							ftripletlosses[l] = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
							triplet_loss2(fanchor2, fpositive2, fnegative2_1)
						tripletlosses = tripletlosses.mean()
						ftripletlosses = ftripletlosses.mean()

				# single return = standard approach.
				# usually use it
				else:
					label = label.T[0]
					pred_scores = pred_scores + pred.flatten().cpu().tolist()
					gt_scores = gt_scores + label.cpu().tolist()
					loss_qa = self.l1_loss(pred.flatten(), label.float().detach())
					loss_qa2 = self.l1_loss(pred2.flatten(), label.float().detach())

					# =============================================================================
					# =============================================================================

					indexlabel = torch.argsort(label) # small--> large
					anchor1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0) # d_min
					positive1 = torch.unsqueeze(pred[indexlabel[1],...].contiguous(),dim=0) # d'_min+
					negative1_1 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0) # d_max+

					anchor2 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0)# d_max
					positive2 = torch.unsqueeze(pred[indexlabel[-2],...].contiguous(),dim=0)# d'_max+
					negative2_1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0)# d_min+

					# =============================================================================
					# =============================================================================

					fanchor1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)
					fpositive1 = torch.unsqueeze(pred2[indexlabel[1],...].contiguous(),dim=0)
					fnegative1_1 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)

					fanchor2 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)
					fpositive2 = torch.unsqueeze(pred2[indexlabel[-2],...].contiguous(),dim=0)
					fnegative2_1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)

					assert (label[indexlabel[-1]]-label[indexlabel[1]])>=0
					assert (label[indexlabel[-2]]-label[indexlabel[0]])>=0
					triplet_loss1 = nn.TripletMarginLoss(margin=(label[indexlabel[-1]]-label[indexlabel[1]]), p=1) # d_min,d'_min,d_max
					triplet_loss2 = nn.TripletMarginLoss(margin=(label[indexlabel[-2]]-label[indexlabel[0]]), p=1)

					tripletlosses = triplet_loss1(anchor1, positive1, negative1_1) + \
					triplet_loss2(anchor2, positive2, negative2_1)
					ftripletlosses = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
					triplet_loss2(fanchor2, fpositive2, fnegative2_1)

				# =============================================================================
				# =============================================================================

				# mind blowing TReS loss
				loss = loss_qa + closs + loss_qa2 + closs2 + 0.5*( self.l1_loss(tripletlosses,ftripletlosses.detach())+ self.l1_loss(ftripletlosses,tripletlosses.detach()))+0.05*(tripletlosses+ftripletlosses)

				
				epoch_loss.append(loss.item())
				loss.backward()
				self.solver.step()
				

			# calculate train metrics: Spearman's rank correlation and loss
			train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
			train_loss = sum(epoch_loss) / len(epoch_loss)

			# log train metric
			train_results[epochnum] = (train_loss, train_srcc)
			with open(trainPerformPath, "r+") as file:
				data = json.load(file)
				data.update(train_results)
				file.seek(0)
				json.dump(data, file)

			# validation
			test_srcc, test_plcc = self.test(self.test_data,epochnum,svPath,seed)

			# log val metric
			results[epochnum]=(test_srcc, test_plcc)
			with open(performPath, "r+") as file:
				data = json.load(file)
				data.update(results)
				file.seek(0)
				json.dump(data, file)
			

			# save best model's parameters according to the val's Spearman's correlation
			if test_srcc > best_srcc:
				modelPathbest = svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion),str(seed))
				torch.save(self.net.state_dict(), modelPathbest)
				# update best metrics
				best_srcc = test_srcc
				best_plcc = test_plcc

			print('{}\t{:4.3f}\t\t{:4.4f}\t\t{:4.4f}\t\t{:4.3f}\t\t{}\t\t{:4.3f}'.format(epochnum + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc,self.paras[0]['lr'],self.droplr ))

			# scheduler step
			self.scheduler.step()

			# cosine scheduler dump
			if self.config.scheduler == "cosine" and self.config.dump_cosine > 0:
				if (epochnum+1) % self.config.T_max == 0:
					self.scheduler.eta_min = self.scheduler.eta_min * self.config.dump_cosine
				if (epochnum+1+self.config.T_max) % self.config.T_max == 0:
					self.scheduler.base_lrs[0] = self.scheduler.base_lrs[0] * self.config.dump_cosine

		print('Best val SRCC %f, PLCC %f' % (best_srcc, best_plcc))

		return best_srcc, best_plcc

	def test(self, data,epochnum,svPath,seed,pretrained=0):
		# to handle test session (in opposite to val session)
		# set pretrained=1 if want test session
		if pretrained:
			self.net.load_state_dict(torch.load(svPath+'/bestmodel_{}_{}'.format(str(self.config.vesion),str(seed))))

		self.net.eval()
		pred_scores = []
		gt_scores = []
		
		pbartest = tqdm(data, leave=False)
		with torch.no_grad():
			steps2 = 0
	
			for img, label, info in pbartest:
				img = torch.as_tensor(img.to(self.device))
				label = torch.as_tensor(label.to(self.device))
				info = torch.as_tensor(info.to(self.device))

				# if we use fuser that processes labels
				# labels must be transferred as the input as well
				# ! MUST NOT USE 0-th label because it's orginal label !
				# ! DON'T CONFUSE WITH NOT TAKING 0-th label at the preprocess stage !

				pred,_ = self.net(img, label[::, 1:self.config.k_late + 1], info)

				if self.config.multi_return:
					pred_scores = pred_scores + pred[:,0].flatten().cpu().tolist()
					gt_scores = gt_scores + label[:,0].flatten().cpu().tolist()
				else:
					pred_scores = pred_scores + pred.flatten().cpu().tolist()
					gt_scores = gt_scores + label.T[0].cpu().tolist()
						
				steps2 += 1
				
		# average scores over patсhes		
		pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
		gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
		
		# if val session, save to val csv
		if not pretrained:
			dataPath = svPath + '/val_prediction_gt_{}_{}_{}.csv'.format(str(self.config.vesion),str(seed),epochnum)
			with open(dataPath, 'w') as f:
				writer = csv.writer(f)
				writer.writerow(g for g in ['preds','gts'])
				writer.writerows(zip(pred_scores, gt_scores))
		# if test session, save to test csv
		else:
			dataPath = svPath + '/test_prediction_gt_{}_{}_{}.csv'.format(str(self.config.vesion),str(seed),epochnum)
			with open(dataPath, 'w') as f:
				writer = csv.writer(f)
				writer.writerow(g for g in ['preds','gts'])
				writer.writerows(zip(pred_scores, gt_scores))
			
		# calculate test metrics: Spearman's and Pearson's correlations	
		test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
		test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
		return test_srcc, test_plcc
	
if __name__=='__main__':
	import os
	import argparse
	import random
	import numpy as np
	from args import *
	
	