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
    def __init__(self, dpm_checkpoints, train_dataset, cuda=-1, K=9):
        super(RetIQANet, self).__init__()
        # define number of neibours
        self.K = K
        # define content perception module
        self.spm = models.vgg16_bn(pretrained=True).features
        if cuda>-1:
            self.spm = self.spm.to(f"cuda:{cuda}")
        # define distortion perception module
        dpm = botnet(dpm_checkpoints, resolution=(288, 384), heads=16, num_classes=125)
        self.dpm = torch.nn.Sequential(*list(dpm.children())[:-2])
        # if config.is_freeze_dpm:
        for p in self.dpm.parameters():
            p.requires_grad = False
        if cuda>-1:
            self.dpm = self.dpm.to(f"cuda:{cuda}")

        # create index database from train_dataset
        ret_db = {
            "ref_path": [],
            "pic_path": [],
            "metric": [],
            "vgg16_path": []
        }

        r_p = []
        sem_db = {
            "ref_path": []
        }

        semantic_features = []

        for _, _ ,y in train_dataset:
            ret_db['ref_path'].append(y['ref_path'][0])
            ret_db['pic_path'].append(y['pic_path'][0])
            ret_db['metric'].append(y['metric'][0])
            ret_db['vgg16_path'].append(y['vgg16_path'][0])

            if y['ref_path'][0] not in r_p:

                sem_db['ref_path'].append(y['ref_path'][0])

                vgg16_feature_tensor = torch.load(y['vgg16_path'][0])
                semantic_features.append(vgg16_feature_tensor.reshape([1, -1]))
                r_p.append(y['ref_path'][0])

        self.ret_db = pd.DataFrame(ret_db)
        self.sem_db = pd.DataFrame(sem_db)
        semantic_features = torch.concat(semantic_features)
        # normed semantics features for cos sim calculation
        self.semantic_features = semantic_features/vector_norm(semantic_features, dim=-1).reshape([semantic_features.shape[0], -1])

        print(len(self.ret_db), len(self.sem_db), self.semantic_features.shape)

    def forward(self, ycbcr, rgb):

        # calculate semantic features of given rgb images
        f_content = self.spm(rgb).reshape([rgb.shape[0], -1])
        # normalize semantic features for cos sim calculation
        f_content = f_content/vector_norm(f_content, dim=-1).reshape([f_content.shape[0], -1])

        # matrix of scalar products of normed vectors = matrix of cosines
        sem_cos = f_content@self.semantic_features.T
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
                all_distorted_image_paths_n_metrics_for_current_ref = self.ret_db.loc[self.ret_db["ref_path"] == current_ref, ["pic_path", "metric"]]
                all_distorted_images_for_current_ref = []
                metrics = []
                ycbcr_transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.448, 0.483, 0.491),
                        std=(0.248, 0.114, 0.106)
                    ),
                    T.RandomCrop((288, 384))
                ])

                for line in all_distorted_image_paths_n_metrics_for_current_ref.iloc:
                    metrics.append(line[1])
                    image = Image.open(line[0]).convert('YCbCr')
                    tensor = ycbcr_transform(image)
                    all_distorted_images_for_current_ref.append(tensor)

                # stack image-tensors to feed into dpm
                all_distorted_images_for_current_ref = torch.stack(all_distorted_images_for_current_ref).to(rgb.device)
                # feed distorted images with the nearest semantics to dpm and reshape
                all_distortion_fs_for_current_ref = self.dpm(all_distorted_images_for_current_ref).reshape([all_distorted_images_for_current_ref.shape[0], -1])
                # normalize
                all_distortion_fs_for_current_ref = all_distortion_fs_for_current_ref/vector_norm(all_distortion_fs_for_current_ref, dim=-1).reshape([all_distortion_fs_for_current_ref.shape[0], -1])
                # matrix of scalar products of normed vectors = matrix of cosines
                dst_cos = (f_distorsion[i].unsqueeze(0))@all_distortion_fs_for_current_ref.T
                # take top-K indeces
                sorted_dst_cos = dst_cos.argsort(dim=-1, descending=True)[::,:1]
                result.append(metrics[sorted_dst_cos.item()].item())

            batch_results.append(statistics.mean(result))
                
        return batch_results


        