import preprocess
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("/home/sharfikeg/my_files/retIQA/dc_ret/DistorsionFeatureExtractor/TReS")
import torch
import torchvision.transforms as T
import argparse
from TReS.models import Net
import pandas as pd
from PIL import Image
import random
from scipy import stats
from config import load_config

torch.set_num_threads(4)
torch.multiprocessing.set_sharing_strategy('file_system')

tres_save_path = "/home/sharfikeg/my_files/extra_disk_1/Save_TReS/"

def main(args):
    # check PURE TRES on BIQ
    dataset_config = argparse.Namespace()
    dataset_config.dataset = args.dataset
    dataset_config.data_path = args.data_path
    dataset_config.ref_path = args.ref_path
    dataset_config.csv_path = args.csv_path
    dataset_config.patches = 50
    dataset_config.batch_size2 = args.batch_size2
    dataset_config.batch_size = args.batch_size
    dataset_config.num_workers = 12
    dataset_config.uni = args.uni

    backbone_config = argparse.Namespace()
    backbone_config.network = 'resnet50'
    backbone_config.nheadt = 16
    backbone_config.num_encoder_layerst = 2
    backbone_config.dim_feedforwardt = 64
    backbone_config.device = args.backbone_device

    to_add = {
        "seed":[],
        "srocc":[],
        "plcc":[],
        "patches": []
    }
    table = pd.DataFrame(to_add)
    sroccs=[]
    plccs=[]
    seeds=[]

    for seed in range(1, args.num_seeds+1):
        dataset_config.seed = seed
        torch.manual_seed(dataset_config.seed)
        torch.cuda.manual_seed(dataset_config.seed)
        np.random.seed(dataset_config.seed)
        random.seed(dataset_config.seed)

        backbone_config.ckpt = tres_save_path + f"{dataset_config.dataset}_1_{seed}/sv/bestmodel_1_{seed}"
        _, _, test_loader  = preprocess.load_train_test_val_data(dataset_config)


        tres = Net(backbone_config, backbone_config.device).to(backbone_config.device)
        tres.load_state_dict(torch.load(backbone_config.ckpt))
        tres.eval()

        r_s = []
        gr_trs = []
        for _, _, rgb_2, y in test_loader:
            rgb_2 = rgb_2.to(backbone_config.device)

            res = tres(rgb_2.reshape([rgb_2.shape[0]*rgb_2.shape[1], 3, 224, 224 ])
                )[0].reshape([rgb_2.shape[0], rgb_2.shape[1]]).mean(dim=1).flatten().detach().cpu()

            gr_trs.append(y['metric'].cpu())
            r_s.append(res)

        srocc = stats.spearmanr(torch.concat(r_s), torch.concat(gr_trs))[0]
        plcc = stats.pearsonr(torch.concat(r_s), torch.concat(gr_trs))[0]
        print(f"PURE TRES on {args.dataset}: SROCC={srocc}, PLCC={plcc}, seed={seed}")
        seeds.append(dataset_config.seed)
        sroccs.append(srocc)
        plccs.append(plcc)
    table['seed'] = seeds
    table["srocc"] = sroccs
    table["plcc"] = plccs
    table["patches"] = [50 for _ in range(args.num_seeds)]

    table.loc["mean"] = table.mean()
    table.loc["median"] = table.median()
    table.loc["std"] = table.std()

    table.to_csv(f"/home/sharfikeg/my_files/retIQA/dc_ret/all_results/{dataset_config.dataset}_pure_tres_test.csv")

if __name__ == '__main__':
    args = load_config()
    main(args)
