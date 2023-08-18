import torch
from torch import nn
from torch.linalg import vector_norm
from torchvision import models
import torchvision.transforms as T
from BoTNet import botnet
import os
import pandas as pd
from PIL import Image
import statistics   



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
        dpm = botnet(dpm_checkpoints, resolution=(288, 384), heads=16, num_classes=num_classes)
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

    def forward(self, ycbcr, rgb):

        # calculate semantic features of given rgb images
        f_content = self.spm(rgb).reshape([rgb.shape[0], -1])
        # normalize semantic features for cos sim calculation
        f_content = f_content/vector_norm(f_content, dim=-1).reshape([f_content.shape[0], -1])

        # matrix of scalar products of normed vectors = matrix of cosines
        sem_cos = f_content@self.semantic_features.T.to(f_content.device)
        # take top-K indeces
        sorted_sem_cos = sem_cos.argsort(dim=-1, descending=True)[::,:self.K]

        

        # calculate distorsion features of given ycbcr images
        f_distorsion = self.dpm(ycbcr).reshape([rgb.shape[0], -1])
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
                all_distortion_fs_for_current_ref = torch.stack(all_distortion_fs_for_current_ref).to(device=rgb.device)
                # normalize
                all_distortion_fs_for_current_ref = all_distortion_fs_for_current_ref/vector_norm(all_distortion_fs_for_current_ref, dim=-1).reshape([all_distortion_fs_for_current_ref.shape[0], -1])
                # matrix of scalar products of normed vectors = matrix of cosines
                dst_cos = (f_distorsion[i].unsqueeze(0))@all_distortion_fs_for_current_ref.T
                # take top-K indeces
                sorted_dst_cos = dst_cos.argsort(dim=-1, descending=True)[::,:1]
                result.append(metrics[sorted_dst_cos.item()])

            batch_results.append(statistics.mean(result))
                
        return batch_results