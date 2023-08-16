import torch
from torch import nn
from torch.linalg import vector_norm
from torchvision import models
import torchvision.transforms as T
from BoTNet import botnet
from RetIQANet import RetIQANet
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from scipy import stats
import preprocess
import time

torch.set_num_threads(24)

from config import load_config

def load_dataset(args):
    df = pd.read_csv(args.csv_path)
    ycbcr_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=(0.448, 0.483, 0.491),
            std=(0.248, 0.114, 0.106)
        ),
        T.RandomCrop((288, 384))
    ])
    rgb_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    if args.dataset == "csiq":
        train, test = train_test_split(df['image'].unique(), test_size=0.2)
        train_dataset = preprocess.CSIQ(
            refs=train,
            img_dir=args.img_dir,
            ref_dir=args.ref_dir,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=T.ToTensor()
        )
        test_dataset = preprocess.CSIQ(
            refs=test,
            img_dir=args.img_dir,
            ref_dir=args.ref_dir,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=rgb_transform
        )

    elif args.dataset == "kadid10k":
        train, test = train_test_split(df['reference'].unique(), test_size=0.2)
        train_dataset = preprocess.DistortedKadid10k(
            refs=train,
            img_dir=args.img_dir,
            ref_dir=args.ref_dir,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=T.ToTensor()
        )
        test_dataset = preprocess.DistortedKadid10k(
            refs=test,
            img_dir=args.img_dir,
            ref_dir=args.ref_dir,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=rgb_transform
        )

    elif args.dataset == "tid2013":
        train, test = train_test_split(df['ref'].unique(), test_size=0.2)
        train_dataset = preprocess.DistortedTid2013(
            refs=train,
            img_dir=args.img_dir,
            ref_dir=args.ref_dir,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=T.ToTensor()
        )
        test_dataset = preprocess.DistortedTid2013(
            refs=test,
            img_dir=args.img_dir,
            ref_dir=args.ref_dir,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=rgb_transform
        )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader

def main(args):
    train_loader, test_loader = load_dataset(args)
    
    model = RetIQANet(
        dpm_checkpoints=args.finetune_checkpoint,
        train_dataset=train_loader,
        cuda=args.cuda,
        K=args.k
    )

    r_s = []
    gr_trs = []
    if args.cuda > -1:
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"

    for ycbcr, rgb, y in test_loader:
        res = model(ycbcr.to(device), rgb.to(device))
        gr_trs.append(y['metric'])
        r_s.append(res)
    t_r_s = []
    for r in r_s:
        t_r_s.append(torch.tensor(r))
    srocc = stats.spearmanr(torch.concat(t_r_s), torch.concat(gr_trs))[0]
    plcc = stats.pearsonr(torch.concat(t_r_s), torch.concat(gr_trs))[0]

    df = pd.read_csv(args.logging_path, index_col=0)
    results = [
        time.time(),
        args.dataset,
        args.aggregation,
        args.k,
        args.batch_size,
        srocc,
        plcc
    ]
    df.loc[len(df)] = results
    df.to_csv(args.logging_path, index=False)


if __name__ == '__main__':
    args = load_config()
    main(args)