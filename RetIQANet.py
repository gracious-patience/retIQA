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
    def __init__(self, dpm_checkpoints, num_classes, train_dataset, cuda=-1, K=9):
        super(RetIQANet, self).__init__()
        # define number of neibours
        self.K = K
        # define content perception module
        self.spm = models.vgg16_bn(weights='DEFAULT').features
        if cuda>-1:
            self.spm = self.spm.to(f"cuda:{cuda}")
        self.spm = self.spm.eval()
        
        # define distortion perception module
        if dpm_checkpoints == my_botnet_pretrain:
            dpm = ResNet50(resolution=(288, 384), heads=16, num_classes=num_classes)
            dpm.fc[1] = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
            checkpoint = torch.load(dpm_checkpoints)
            try:
                # model = torch.nn.DataParallel(model)
                dpm.load_state_dict(checkpoint['state_dict'],strict=True)
            except:
                dpm.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        else:
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
    def __init__(self, dpm_checkpoints, num_classes, train_dataset, device="cpu", K=9, weighted=False):
        super(NoRefRetIQANet, self).__init__()
        # define number of neighbours
        self.K = K
        # define content perception module
        self.spm = models.vgg16_bn(weights='DEFAULT').features
        self.spm = self.spm.to(device)
        for p in self.spm.parameters():
            p.requires_grad = False
        self.spm = self.spm.eval()
        
        # define distortion perception module
        if dpm_checkpoints == my_botnet_pretrain:
            dpm = ResNet50(resolution=(288, 384), heads=16, num_classes=num_classes)
            dpm.fc[1] = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
            checkpoint = torch.load(dpm_checkpoints, map_location=device)
            try:
                # model = torch.nn.DataParallel(model)
                dpm.load_state_dict(checkpoint['state_dict'],strict=True)
            except:
                dpm.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        else:
            dpm = botnet(pretrained_model=dpm_checkpoints, num_classes=num_classes, resolution=(288, 384), heads=16)
        self.dpm = torch.nn.Sequential(*list(dpm.children())[:-2])
        # if config.is_freeze_dpm:
        for p in self.dpm.parameters():
            p.requires_grad = False
        self.dpm.eval()
        self.dpm = self.dpm.to(device)

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
            ret_db['metric'] += y['metric'].cpu().float().tolist()
            ret_db['vgg16_path'] += y['vgg16_path']

            # save dst_features of each pic
            dst_features = self.dpm(pic.to(device)).reshape([pic.shape[0], -1]).cpu()
            distorsion_features.append(dst_features)

            for k, vgg_path in enumerate(y['vgg16_path']):

                vgg16_feature_tensor = torch.load(y['vgg16_path'][k], map_location=device)
                semantic_features.append(vgg16_feature_tensor.reshape([1, -1]))

        self.ret_db = pd.DataFrame(ret_db)
        semantic_features = torch.concat(semantic_features)
        distorsion_features = torch.concat(distorsion_features)
        # normed semantics features for cos sim calculation
        self.semantic_features = (semantic_features/vector_norm(semantic_features, dim=-1).reshape([semantic_features.shape[0], -1]))
        # normed distorsion features for cos sim calculation
        self.distorsion_features = (distorsion_features/vector_norm(distorsion_features, dim=-1).reshape([distorsion_features.shape[0], -1])).to(device)

        self.weighted = weighted

    def forward(self, ycbcr, rgb_1):
        # calculate semantic features of given rgb images
        f_content = self.spm(rgb_1).reshape([rgb_1.shape[0], -1])
        # normalize semantic features for cos sim calculation
        f_content = f_content/vector_norm(f_content, dim=-1).reshape([f_content.shape[0], -1])

        # matrix of scalar products of normed vectors = matrix of cosines
        sem_cos = f_content@self.semantic_features.T
        # take top-K indeces
        values_sem, sorted_sem_cos = sem_cos.sort(dim=-1, descending=True)
        values_sem, sorted_sem_cos = values_sem[::,:self.K], sorted_sem_cos[::,:self.K]

        # calculate distorsion features of given ycbcr images
        f_distorsion = self.dpm(ycbcr).reshape([rgb_1.shape[0], -1])
        # normalize distorsion features for cos sim calculation
        f_distorsion = f_distorsion/vector_norm(f_distorsion, dim=-1).reshape([f_distorsion.shape[0], -1])

        # matrix of scalar products of normed vectors = matrix of cosines
        dst_cos = f_distorsion@self.distorsion_features.T
        # take top-K indeces
        values_dst, sorted_dst_cos = dst_cos.sort(dim=-1, descending=True)
        values_dst, sorted_dst_cos = values_dst[::,:self.K], sorted_dst_cos[::,:self.K]

        result = []
        retrieved_result = []
        # ret_sems = []
        # ret_dsts = []
        for j in range(f_distorsion.shape[0]):
            small_res = []
            # small_ret_sems = []
            # small_ret_dsts = []
            for m in range(self.K):
                if self.weighted:
                    # print(f"Info in sem: weight={values_sem[j][m].item()}, score={self.ret_db.loc[sorted_sem_cos[j][m].item()]['metric']}, multiplied={values_sem[j][m].item() * self.ret_db.loc[sorted_sem_cos[j][m].item()]['metric'] }")
                    small_res.append(values_sem[j][m].item()/values_sem[j].sum().item() * self.ret_db.loc[sorted_sem_cos[j][m].item()]['metric'] )
                    small_res.append(values_dst[j][m].item()/values_dst[j].sum().item() * self.ret_db.loc[sorted_dst_cos[j][m].item()]['metric'] )
                    # print(f"Info in dst: weight={values_dst[j][m].item()}, score={self.ret_db.loc[sorted_dst_cos[j][m].item()]['metric']}, multiplied={values_dst[j][m].item() * self.ret_db.loc[sorted_dst_cos[j][m].item()]['metric'] }")
                else:
                    small_res.append(self.ret_db.loc[sorted_sem_cos[j][m].item()]['metric'] )
                    small_res.append(self.ret_db.loc[sorted_dst_cos[j][m].item()]['metric'] )
                # small_ret_sems.append(self.semantic_features[sorted_sem_cos[j][m].item()])
                # small_ret_dsts.append(self.distorsion_features[sorted_dst_cos[j][m].item()])
            result.append(statistics.mean(small_res))
            retrieved_result.append(torch.tensor(small_res))
            # ret_sems.append(torch.stack(small_ret_sems))
            # ret_dsts.append(torch.stack(small_ret_dsts))

        return torch.tensor(result) , torch.stack(retrieved_result) #, f_content, f_distorsion, torch.stack(ret_sems), torch.stack(ret_dsts)

class ConcatNoRefRetIQANet(nn.Module):
    def __init__(self, dpm_checkpoints, num_classes, train_dataset, device="cpu", K=9, weighted=True):
        super(ConcatNoRefRetIQANet, self).__init__()
        # define number of neighbours
        self.K = K
        # define content perception module
        self.spm = models.vgg16_bn(weights='DEFAULT').features
        self.spm = self.spm.to(device)
        for p in self.spm.parameters():
            p.requires_grad = False
        self.spm = self.spm.eval()
        
        # define distortion perception module
        if dpm_checkpoints == my_botnet_pretrain:
            dpm = ResNet50(resolution=(288, 384), heads=16, num_classes=num_classes)
            dpm.fc[1] = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
            checkpoint = torch.load(dpm_checkpoints, map_location=device)
            try:
                # model = torch.nn.DataParallel(model)
                dpm.load_state_dict(checkpoint['state_dict'],strict=True)
            except:
                dpm.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        else:
            dpm = botnet(pretrained_model=dpm_checkpoints, num_classes=num_classes, resolution=(288, 384), heads=16)
        self.dpm = torch.nn.Sequential(*list(dpm.children())[:-2])
        # if config.is_freeze_dpm:
        for p in self.dpm.parameters():
            p.requires_grad = False
        self.dpm.eval()
        self.dpm = self.dpm.to(device)

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
            ret_db['metric'] += y['metric'].cpu().float().tolist()
            ret_db['vgg16_path'] += y['vgg16_path']

            # save dst_features of each pic
            dst_features = self.dpm(pic.to(device)).reshape([pic.shape[0], -1]).cpu()
            distorsion_features.append(dst_features)

            for k, vgg_path in enumerate(y['vgg16_path']):

                vgg16_feature_tensor = torch.load(y['vgg16_path'][k], map_location=device)
                semantic_features.append(vgg16_feature_tensor.reshape([1, -1]))

        self.ret_db = pd.DataFrame(ret_db)
        semantic_features = torch.concat(semantic_features)
        distorsion_features = torch.concat(distorsion_features)

        # normed concated features for cos sim calculation
        self.concated_features = torch.concat(
            [
                (semantic_features/vector_norm(semantic_features, dim=-1).reshape([semantic_features.shape[0], -1])),
                (distorsion_features/vector_norm(distorsion_features, dim=-1).reshape([distorsion_features.shape[0], -1])).to(device)
            ], dim=-1
        ) 
        self.weighted = weighted

    def forward(self, ycbcr, rgb_1):
        # calculate semantic features of given rgb images
        f_content = self.spm(rgb_1).reshape([rgb_1.shape[0], -1])
        # normalize semantic features for cos sim calculation
        f_content = f_content/vector_norm(f_content, dim=-1).reshape([f_content.shape[0], -1])
        # calculate distorsion features of given ycbcr images
        f_distorsion = self.dpm(ycbcr).reshape([rgb_1.shape[0], -1])
        # normalize distorsion features for cos sim calculation
        f_distorsion = f_distorsion/vector_norm(f_distorsion, dim=-1).reshape([f_distorsion.shape[0], -1])
        # concat sem and dst features
        f_concat = torch.concat(
            [f_content, f_distorsion], dim=1
        )


        # matrix of scalar products of normed vectors = matrix of cosines
        sim_cos = f_concat@self.concated_features.T
        # take top-K indeces
        values, indices = sim_cos.sort(dim=-1, descending=True)
        values, indices = values[::,:self.K], indices[::,:self.K]

       
        result = []
        retrieved_result = []
        # ret_sems = []
        # ret_dsts = []
        for j in range(f_concat.shape[0]):
            small_res = []
            # small_ret = []
            for m in range(self.K):
                if self.weighted:
                    # print(f"Info in sem: weight={values_sem[j][m].item()}, score={self.ret_db.loc[sorted_sem_cos[j][m].item()]['metric']}, multiplied={values_sem[j][m].item() * self.ret_db.loc[sorted_sem_cos[j][m].item()]['metric'] }")
                    small_res.append(values[j][m].item()/values[j].sum().item() * self.ret_db.loc[indices[j][m].item()]['metric'] )
                    # print(f"Info in dst: weight={values_dst[j][m].item()}, score={self.ret_db.loc[sorted_dst_cos[j][m].item()]['metric']}, multiplied={values_dst[j][m].item() * self.ret_db.loc[sorted_dst_cos[j][m].item()]['metric'] }")
                else:
                    small_res.append(self.ret_db.loc[indices[j][m].item()]['metric'] )
                    # small_ret.append(self.semantic_features[indices[j][m].item()])
            result.append(statistics.mean(small_res))
            retrieved_result.append(torch.tensor(small_res))
            # ret_sems.append(torch.stack(small_ret_sems))
            # ret_dsts.append(torch.stack(small_ret_dsts))

        return torch.tensor(result) , torch.stack(retrieved_result) #, f_content, f_distorsion, torch.stack(ret_sems), torch.stack(ret_dsts)


def SimpleAggro(x, y):
    to_fuse = torch.stack([
        x,
        y
    ])   
    return to_fuse.mean(dim=0)

def ConcatAggro(back_bone, k_retrieved):
    # print(back_bone.unsqueeze(0).shape, k_retrieved.shape )
    to_fuse = torch.concat([
        back_bone.unsqueeze(0).T,
        k_retrieved
    ], dim=1)
    return to_fuse.mean(dim=1).flatten()

def LinearCombination(a, x, b, y):
    return a*x + b*y

class MAM(nn.Module):
    def __init__(self, embed_dim, vdim, L):
        super(MAM, self).__init__()
        self.xi_1 = nn.Linear(
            in_features = vdim,
            out_features = embed_dim,
            bias=False
        )
        self.xi_2 = nn.Linear(
            in_features = vdim,
            out_features = embed_dim,
            bias=False
        )
        self.softmax = nn.Softmax(dim=0)
        self.embed_dim = embed_dim
        self.L = L
    
    def forward(self, q, k, v, b_score):
        cur = z = q

        for _ in range(self.L):
            cur = torch.tensordot(
                cur,
                k.permute((2,1,0)),
                dims=1
            )
            cur = self.softmax(cur/torch.sqrt(torch.tensor(self.embed_dim)))
            cur = self.xi_1(
                torch.tensordot(
                    cur,
                    v,
                    dims=([1, 2], [1, 0])
                )
            )
            cur = self.xi_2(b_score) + z
        return cur
    
class Regressor(nn.Module):
    def __init__(self, embed_dim):
        super(Regressor, self).__init__()
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(
            in_features=embed_dim,
            out_features=1
        )

    def forward(self, x):
        x = self.linear1(self.tanh(x))
        return x

def PosEncode(x, dim):
    embed = torch.zeros([dim])
    for i in range(0, dim, 2):
        embed[i] = torch.sin(x/(10000**(2*i/dim)))
        embed[i+1] = torch.cos(x/(10000**(2*i/dim)))
    return embed
    
class AkimboNet(nn.Module):
    def __init__(self, ret_config, backbone_config, setup="no_reference"):
        super(AkimboNet, self).__init__()

        self.ret_config = ret_config
        self.backbone_config = backbone_config

        # retrieval module
        if setup == "reference":
            self.ret_module = RetIQANet(
                ret_config.dpm_checkpoints,
                ret_config.num_classes,
                ret_config.train_dataset,
                ret_config.device,
                ret_config.K
            )
        elif setup == "concat_no_reference":
            self.ret_module = ConcatNoRefRetIQANet(
                ret_config.dpm_checkpoints,
                ret_config.num_classes,
                ret_config.train_dataset,
                ret_config.device,
                ret_config.K,
                ret_config.weighted
            )
        elif setup == "no_reference":
            self.ret_module = NoRefRetIQANet(
                ret_config.dpm_checkpoints,
                ret_config.num_classes,
                ret_config.train_dataset,
                ret_config.device,
                ret_config.K,
                ret_config.weighted
            )

        # main backbone
        self.backbone_module = Net(backbone_config, backbone_config.device).to(backbone_config.device)
        self.backbone_module.load_state_dict(torch.load(backbone_config.ckpt, map_location=backbone_config.device))
        for param in self.backbone_module.parameters():
            param.requires_grad = False
        self.backbone_module.eval()

        # aggregator module
        self.aggregator_module = SimpleAggro

    def forward(self, ycbcr, rgb_1, rgb_2, a=0.5, b=0.5):
        # main backbone score
        backbone_score = self.backbone_module(
            rgb_2.reshape([rgb_2.shape[0]*rgb_2.shape[1], 3, 224, 224 ])
        )[0].reshape([rgb_2.shape[0], rgb_2.shape[1]]).mean(dim=1).flatten().cpu()

        # retrieved score
        averaged_score, retrieval_score = self.ret_module(ycbcr, rgb_1)
        # , retrieval_score, sem_features, dst_features, sem_ret_features, dst_ret_features  = self.ret_module(ycbcr, rgb_1)
        
        # aggregation of backbone and retrieved scores
        # result = LinearCombination(a, backbone_score, b, averaged_score)
        if self.backbone_config.aggregation == "low_tres_impact_averaging":
            result = ConcatAggro(backbone_score, retrieval_score)
        elif self.backbone_config.aggregation == "averaging":
            result = self.aggregator_module(backbone_score, averaged_score)

        return result
        # return result, (backbone_score, retrieval_score, sem_features, dst_features, sem_ret_features, dst_ret_features)