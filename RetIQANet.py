import torch
from torch import nn
from torch.linalg import vector_norm
from torchvision import models
import torchvision.transforms as T
from BoTNet import botnet
from model import ResNet50
from TReS.models import Net
import os
import pandas as pd
from PIL import Image
import statistics   


my_botnet_pretrain = "/home/sharfikeg/my_files/retIQA/dc_ret/my_botnet_pretrain/checkpoint_model_best_heads16.pth"
botnet_pretrain="/home/sharfikeg/my_files/VIPNet/pretrained_model/botnet_model_best.pth.tar"

class RetIQANet(nn.Module):
    def __init__(self, dpm_checkpoints, num_classes, train_dataset, cuda=-1, K=9, my=True):
        super(RetIQANet, self).__init__()
        # define number of neibours
        self.K = K
        # define content perception module
        self.spm = models.vgg16_bn(weights='DEFAULT').features
        if cuda>-1:
            self.spm = self.spm.to(f"cuda:{cuda}")
        self.spm = self.spm.eval()
        
        # define distortion perception module
        if my:
            dpm = ResNet50(resolution=(288, 384), heads=16, num_classes=num_classes)
            # dpm.fc[1] = nn.Linear(in_features=2048, out_features=125, bias=True)
            checkpoint = torch.load(dpm_checkpoints)
            try:
                # model = torch.nn.DataParallel(model)
                dpm.load_state_dict(checkpoint['state_dict'],strict=True)
            except:
                dpm.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        elif not my:
            dpm = botnet(pretrained_model=dpm_checkpoints, num_classes=num_classes, resolution=(288, 384), heads=16)
        self.dpm = torch.nn.Sequential(*list(dpm.children())[:-2])
        # if config.is_freeze_dpm:
        for p in self.dpm.parameters():
            p.requires_grad = False
        self.dpm.eval()
        if cuda>-1:
            self.dpm = self.dpm.to(f"cuda:{cuda}")

        # create index database from train_dataset
        ret_db = {
            "ref_path": [],
            "pic_path": [],
            "pic_dst_fs": [],
            "metric": [],
            "vgg16_path": []
        }

        r_p = []
        sem_db = {
            "ref_path": []
        }

        semantic_features = []

        for pic, y in train_dataset:

            ret_db['ref_path'] += y['ref_path']
            ret_db['pic_path'] += y['pic_path']
            ret_db['metric'] += y['metric'].cpu().tolist()
            ret_db['vgg16_path'] += y['vgg16_path']

            # save dst_features of each pic
            dst_features = self.dpm(pic.to(f"cuda:{cuda}")).reshape([pic.shape[0], -1])
            ret_db['pic_dst_fs']+=list(dst_features.cpu())

            for k, ref in enumerate(y['ref_path']):
                if ref not in r_p:

                    sem_db['ref_path'].append(ref)

                    vgg16_feature_tensor = torch.load(y['vgg16_path'][k])
                    semantic_features.append(vgg16_feature_tensor.reshape([1, -1]))
                    r_p.append(ref)

        self.ret_db = pd.DataFrame(ret_db)
        self.sem_db = pd.DataFrame(sem_db)
        semantic_features = torch.concat(semantic_features)
        # normed semantics features for cos sim calculation
        self.semantic_features = (semantic_features/vector_norm(semantic_features, dim=-1).reshape([semantic_features.shape[0], -1])).cpu()

        print(len(self.ret_db), len(self.sem_db), self.semantic_features.shape)

    def forward(self, ycbcr, rgb_1):
        # calculate semantic features of given rgb images
        f_content = self.spm(rgb_1).reshape([rgb_1.shape[0], -1])
        # normalize semantic features for cos sim calculation
        f_content = f_content/vector_norm(f_content, dim=-1).reshape([f_content.shape[0], -1])

        # matrix of scalar products of normed vectors = matrix of cosines
        sem_cos = f_content@self.semantic_features.T.to(f_content.device)
        # take top-K indeces
        sorted_sem_cos = sem_cos.argsort(dim=-1, descending=True)[::,:self.K]

        # calculate distorsion features of given ycbcr images
        f_distorsion = self.dpm(ycbcr).reshape([rgb_1.shape[0], -1])
        # normalize distorsion features for cos sim calculation
        f_distorsion = f_distorsion/vector_norm(f_distorsion, dim=-1).reshape([f_distorsion.shape[0], -1])

        # iterate over batch
        # bad, optimize
        batch_results = []
        for i, row in enumerate(sorted_sem_cos):
            # iterate over refs (nearest semantic pics)
            result = []
            for guy in row:
                current_ref = self.sem_db.iloc[guy.item()]["ref_path"]
                all_distorted_image_paths_n_metrics_for_current_ref = self.ret_db.loc[self.ret_db["ref_path"] == current_ref, ["pic_dst_fs", "metric"]]
              
                metrics = list(all_distorted_image_paths_n_metrics_for_current_ref["metric"])
                all_distortion_fs_for_current_ref = list(all_distorted_image_paths_n_metrics_for_current_ref["pic_dst_fs"])
               
                # concat dst features
                all_distortion_fs_for_current_ref = torch.stack(all_distortion_fs_for_current_ref).to(device=rgb_1.device)
                # normalize
                all_distortion_fs_for_current_ref = all_distortion_fs_for_current_ref/vector_norm(all_distortion_fs_for_current_ref, dim=-1).reshape([all_distortion_fs_for_current_ref.shape[0], -1])
                # matrix of scalar products of normed vectors = matrix of cosines
                dst_cos = (f_distorsion[i].unsqueeze(0))@all_distortion_fs_for_current_ref.T
                # take top-K indeces
                sorted_dst_cos = dst_cos.argsort(dim=-1, descending=True)[::,:1]
                result.append(metrics[sorted_dst_cos.item()])

            batch_results.append(statistics.mean(result))

        return torch.tensor(batch_results)

class NoRefRetIQANet(nn.Module):
    def __init__(self, dpm_checkpoints, num_classes, train_dataset, cuda=-1, K=9, my=True):
        super(NoRefRetIQANet, self).__init__()
        # define number of neibours
        self.K = K
        # define content perception module
        self.spm = models.vgg16_bn(weights='DEFAULT').features
        if cuda>-1:
            self.spm = self.spm.to(f"cuda:{cuda}")
        self.spm = self.spm.eval()
        
        # define distortion perception module
        if my:
            dpm = ResNet50(resolution=(288, 384), heads=16, num_classes=num_classes)
            # dpm.fc[1] = nn.Linear(in_features=2048, out_features=125, bias=True)
            checkpoint = torch.load(dpm_checkpoints)
            try:
                # model = torch.nn.DataParallel(model)
                dpm.load_state_dict(checkpoint['state_dict'],strict=True)
            except:
                dpm.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        elif not my:
            dpm = botnet(pretrained_model=dpm_checkpoints, num_classes=num_classes, resolution=(288, 384), heads=16)
        self.dpm = torch.nn.Sequential(*list(dpm.children())[:-2])
        # if config.is_freeze_dpm:
        for p in self.dpm.parameters():
            p.requires_grad = False
        self.dpm.eval()
        if cuda>-1:
            self.dpm = self.dpm.to(f"cuda:{cuda}")

        # create index database from train_dataset
        ret_db = {
            "pic_path": [],
            "metric": [],
            "vgg16_path": []
        }

        distorsion_features = []
        semantic_features = []

        for pic, y in train_dataset:

            ret_db['pic_path'] += y['pic_path']
            ret_db['metric'] += y['metric'].cpu().tolist()
            ret_db['vgg16_path'] += y['vgg16_path']

            # save dst_features of each pic
            dst_features = self.dpm(pic.to(f"cuda:{cuda}")).reshape([pic.shape[0], -1])
            distorsion_features.append(dst_features)

            for k, vgg_path in enumerate(y['vgg16_path']):

                vgg16_feature_tensor = torch.load(y['vgg16_path'][k])
                semantic_features.append(vgg16_feature_tensor.reshape([1, -1]))

        self.ret_db = pd.DataFrame(ret_db)
        semantic_features = torch.concat(semantic_features).detach().cpu()
        distorsion_features = torch.concat(distorsion_features).detach().cpu()
        # normed semantics features for cos sim calculation
        self.semantic_features = (semantic_features/vector_norm(semantic_features, dim=-1).reshape([semantic_features.shape[0], -1])).cpu()
        # normed distorsion features for cos sim calculation
        self.distorsion_features = (distorsion_features/vector_norm(distorsion_features, dim=-1).reshape([distorsion_features.shape[0], -1])).cpu()

        print(len(self.ret_db), distorsion_features.shape , self.semantic_features.shape)

    def forward(self, ycbcr, rgb_1):
        # calculate semantic features of given rgb images
        f_content = self.spm(rgb_1).reshape([rgb_1.shape[0], -1])
        # normalize semantic features for cos sim calculation
        f_content = f_content/vector_norm(f_content, dim=-1).reshape([f_content.shape[0], -1])

        # matrix of scalar products of normed vectors = matrix of cosines
        sem_cos = f_content@self.semantic_features.T.to(f_content.device)
        # take top-K indeces
        sorted_sem_cos = sem_cos.argsort(dim=-1, descending=True)[::,:self.K]

        # calculate distorsion features of given ycbcr images
        f_distorsion = self.dpm(ycbcr).reshape([rgb_1.shape[0], -1])
        # normalize distorsion features for cos sim calculation
        f_distorsion = f_distorsion/vector_norm(f_distorsion, dim=-1).reshape([f_distorsion.shape[0], -1])

        # matrix of scalar products of normed vectors = matrix of cosines
        dst_cos = f_distorsion@self.distorsion_features.T.to(f_content.device)
        # take top-K indeces
        sorted_dst_cos = dst_cos.argsort(dim=-1, descending=True)[::,:self.K]

        result = []
        for j in range(f_distorsion.shape[0]):
            small_res = []
            for m in range(self.K):
                small_res.append(self.ret_db.loc[sorted_sem_cos[j][m].item()]['metric'] )
                small_res.append(self.ret_db.loc[sorted_dst_cos[j][m].item()]['metric'] )
            result.append(statistics.mean(small_res))

        return torch.tensor(result)

def SimpleAggro(x, y):
    to_fuse = torch.stack([
        x,
        y
    ])   

    return to_fuse.mean(dim=0)
    
class AkimboNet(nn.Module):
    def __init__(self, ret_config, backbone_config, setup):
        super(AkimboNet, self).__init__()


        # retrieval module
        if setup == "reference":
            self.ret_module = RetIQANet(
                ret_config.dpm_checkpoints,
                ret_config.num_classes,
                ret_config.train_dataset,
                ret_config.cuda,
                ret_config.K,
                ret_config.my
            )
        elif setup == "no_reference":
            self.ret_module = NoRefRetIQANet(
                ret_config.dpm_checkpoints,
                ret_config.num_classes,
                ret_config.train_dataset,
                ret_config.cuda,
                ret_config.K
            )

        # main backbone
        self.backbone_module = Net(backbone_config, f"cuda:{backbone_config.cuda}").to(f"cuda:{backbone_config.cuda}")
        self.backbone_module.load_state_dict(torch.load(backbone_config.ckpt))
        self.backbone_module.eval()

        # aggregator module
        self.aggregator_module = SimpleAggro

    def forward(self, ycbcr, rgb_1, rgb_2):
        # main backbone score
        backbone_score = self.backbone_module(
            rgb_2.reshape([rgb_2.shape[0]*rgb_2.shape[1], 3, 224, 224 ])
        )[0].reshape([rgb_2.shape[0], rgb_2.shape[1]]).mean(dim=1).flatten().detach().cpu()

        # retrieved score
        retrieval_score = self.ret_module(ycbcr, rgb_1)

        # aggregation of backbone and retrieved scores
        result = self.aggregator_module(backbone_score, retrieval_score)

        return result