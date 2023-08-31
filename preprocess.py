from typing import Any
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import os
import glob
from PIL import Image
import pandas as pd
import torch

img_num = {
        'csiq':     list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013':  list(range(0, 25)),
        'koniq':    list(range(0, 10073)),
        'spaq':     list(range(0, 11125))
        }



def load_data(args):

    small_transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10('./data', train=True, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_transform = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        ])
        test_dataset = CIFAR10('./data', train=False, transform=test_transform)

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.dataset == 'kadis700k':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_height, args.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.448, 0.483, 0.491],
                std=[0.248, 0.114, 0.106]
            )
            
        ])
        dataset = DistortedKadis700k(args.data_path, transform=train_transform)
        print(len(dataset))

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.dataset == 'kadid10k':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_height, args.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.448, 0.483, 0.491],
                std=[0.248, 0.114, 0.106]
            ) 
        ])

        df = pd.read_csv(args.csv_path)
        train, test = train_test_split(df['reference'].unique()[:-1], test_size=0.2, random_state=args.seed)
        train_dataset = DistortedKadid10k(
            refs=train,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=train_transform,
            rgb_transform=small_transform,
            backbone_transform=transforms.ToTensor(),
            type="test"
        )
        test_dataset = DistortedKadid10k(
            refs=test,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=train_transform,
            rgb_transform=small_transform,
            backbone_transform=transforms.ToTensor(),
            type="test"
        )
        print(len(train_dataset), len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.dataset == 'tid2013':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_height, args.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.448, 0.483, 0.491],
                std=[0.248, 0.114, 0.106]
            ) 
        ])

        df = pd.read_csv(args.csv_path)
        train, test = train_test_split(df['ref'].unique(), test_size=0.2, random_state=args.seed)
        train_dataset = DistortedTid2013(
            refs= train,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=train_transform,
            rgb_transform=small_transform,
            backbone_transform=transforms.ToTensor(),
            type="test"
        )
        test_dataset = DistortedTid2013(
            refs= test,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=train_transform,
            rgb_transform=small_transform,
            backbone_transform=transforms.ToTensor(),
            type="test"
        )
        print(len(train_dataset), len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.dataset == 'csiq':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_height, args.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.448, 0.483, 0.491],
                std=[0.248, 0.114, 0.106]
            )
        ])

        df = pd.read_csv(args.csv_path)
        train, test = train_test_split(df['image'].unique(), test_size=0.2, random_state=args.seed)
        train_dataset = CSIQ(
            refs=train,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=train_transform,
            rgb_transform=small_transform,
            backbone_transform=transforms.ToTensor(),
            type="test"
        )
        test_dataset = CSIQ(
            refs=test,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=train_transform,
            rgb_transform=small_transform,
            backbone_transform=transforms.ToTensor(),
            type="test"
        )
        print(len(train_dataset), len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader



def load_data2(args):
    ycbcr_transform = transforms.Compose([
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.448, 0.483, 0.491),
            std=(0.248, 0.114, 0.106)
        )
    ])
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])
    backbone_transform = transforms.Compose([
        transforms.RandomCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    if args.dataset == "csiq":
        df = pd.read_csv(args.csv_path)
        train, test = train_test_split(df['image'].unique(), test_size=0.2, random_state=args.seed)

        train_dataset = CSIQ(
            refs=train,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=None,
            backbone_transform=None,
            type="train"
        )
        test_dataset = CSIQ(
            refs=test,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=rgb_transform,
            backbone_transform=backbone_transform,
            type="test",
            patches=args.patches
        )
    elif args.dataset == "kadid10k":
        df = pd.read_csv(args.csv_path)
        train, test = train_test_split(df['reference'].unique()[:-1], test_size=0.2, random_state=args.seed)

        train_dataset = DistortedKadid10k(
            refs=train,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=None,
            backbone_transform=None,
            type="train"
        )
        test_dataset = DistortedKadid10k(
            refs=test,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=rgb_transform,
            backbone_transform=backbone_transform,
            type="test",
            patches=args.patches
        )
    elif args.dataset == "tid2013":
        df = pd.read_csv(args.csv_path)
        train, test = train_test_split(df['ref'].unique(), test_size=0.2, random_state=args.seed)

        train_dataset = DistortedTid2013(
            refs=train,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=None,
            backbone_transform=None,
            type="train"
        )
        test_dataset = DistortedTid2013(
            refs=test,
            img_dir=args.data_path,
            ref_dir=args.ref_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=rgb_transform,
            backbone_transform=backbone_transform,
            type="test",
            patches=args.patches
        )
    elif args.dataset == "koniq":
        total_num_images = img_num[args.dataset]
        train_indeces, test_indeces = train_test_split(total_num_images, test_size=0.2, random_state=args.seed)

        train_dataset = Koniq10k(
            indeces=train_indeces,
            img_dir=args.data_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=None,
            backbone_transform=None,
            type="train"
        )
        test_dataset = Koniq10k(
            indeces=test_indeces,
            img_dir=args.data_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=rgb_transform,
            backbone_transform=backbone_transform,
            type="test",
            patches=args.patches
        )
    elif args.dataset == "spaq":
        total_num_images = img_num[args.dataset]
        train_indeces, test_indeces = train_test_split(total_num_images, test_size=0.2, random_state=args.seed)

        train_dataset = Spaq(
            indeces=train_indeces,
            img_dir=args.data_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=None,
            backbone_transform=None,
            type="train"
        )
        test_dataset = Spaq(
            indeces=test_indeces,
            img_dir=args.data_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=rgb_transform,
            backbone_transform=backbone_transform,
            type="test",
            patches=args.patches
        )
    elif args.dataset == "liveitw":
        rgb_transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        train_dataset = Koniq10k(
            img_dir=args.data_path,
            csv_path=args.csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=transforms.ToTensor(),
            backbone_transform=None,
            type="train"
        )
        test_dataset = LiveITW(
            img_dir=args.test_data_path,
            csv_path=args.test_csv_path,
            ycbcr_transform=ycbcr_transform,
            rgb_transform=rgb_transform,
            backbone_transform=backbone_transform,
            type="test",
            patches=args.patches
        )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size2, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader


class DistortedTid2013(Dataset):
    def __init__(self, refs, img_dir, ref_dir, csv_path, ycbcr_transform=None, rgb_transform=None, backbone_transform=None, type="train", patches=1):
        self.img_dir = img_dir
        self.ref_dir = ref_dir
        self.ycbcr_transform = ycbcr_transform
        self.rgb_transform = rgb_transform
        self.backbone_transform = backbone_transform
        self.patches = patches

        self.refs = refs
        df = pd.read_csv(csv_path)
        dfs = [df[df["ref"] == ref] for ref in refs]
        self.df = pd.concat(dfs)
        self.len = len(self.df)
        self.type = type
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_info = self.df.iloc[idx].to_dict()

        # 5 = num of dst levels
        img_info['label'] = (img_info['dst_idx']-1)*5 + img_info['dst_lev']-1
        img_path = f"{self.img_dir}{img_info['image']}"
        img_info['pic_path'] = img_path
        img_info['ref_path'] = f"{self.ref_dir}{img_info['ref']}.BMP"
        img_info['metric'] = img_info['mos']

        ycbcr = Image.open(img_path).convert('YCbCr')
        if self.ycbcr_transform:
            ycbcr = self.ycbcr_transform(ycbcr)

        if self.type == "train":
            return ycbcr, img_info

        rgb = Image.open(img_path)
        if self.rgb_transform:
            rgb_1 = self.rgb_transform(rgb)
        if self.backbone_transform:
            rgb_2 = []
            for _ in range(self.patches):
                rgb_2.append(self.backbone_transform(rgb))
        return ycbcr, rgb_1, torch.stack(rgb_2), img_info
    
class DistortedKadid10k(Dataset):
    def __init__(self, refs, img_dir, ref_dir, csv_path, ycbcr_transform=None, rgb_transform=None, backbone_transform=None, type="train", patches=1):
        self.img_dir = img_dir
        self.ref_dir = ref_dir
        self.ycbcr_transform = ycbcr_transform
        self.rgb_transform = rgb_transform
        self.backbone_transform = backbone_transform

        self.refs = refs
        df = pd.read_csv(csv_path)
        dfs = [df[df["reference"] == ref] for ref in refs]
        self.df = pd.concat(dfs)
        self.len = len(self.df)
        self.type = type
        self.patches = patches
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_info = self.df.iloc[idx].to_dict()

        # 5 = num of dst levels
        img_info['label'] = (img_info['noise']-1)*5 + int(img_info['image'].split('_')[-1].split('.')[0]) -1
        img_path = f"{self.img_dir}{img_info['image']}"
        img_info['pic_path'] = img_path
        img_info['ref_path'] = f"{self.ref_dir}{img_info['reference']}"
        img_info['metric'] = img_info['dmos']

        ycbcr = Image.open(img_path).convert('YCbCr')
        if self.ycbcr_transform:
            ycbcr = self.ycbcr_transform(ycbcr)

        if self.type == "train":
            return ycbcr, img_info

        rgb = Image.open(img_path)
        if self.rgb_transform:
            rgb_1 = self.rgb_transform(rgb)
        if self.backbone_transform:
            rgb_2 = []
            for _ in range(self.patches):
                rgb_2.append(self.backbone_transform(rgb))
        return ycbcr, rgb_1, torch.stack(rgb_2), img_info

class DistortedKadis700k(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(self.img_dir,  '*.bmp'))
        self.length = len(self.img_paths)

    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.split('/')[-1]
        image = Image.open(img_path).convert('YCbCr')
        _ , dist, level = img_name.split('_')

        # 5 = number of distorsion levels
        label = (int(dist)-1)*5 + int(level.split('.')[0])-1
        if self.transform:
            image = self.transform(image)
        return image, label

class CSIQ(Dataset):
    def __init__(self, refs, img_dir, ref_dir, csv_path, ycbcr_transform=None, rgb_transform=None, backbone_transform=None, type="train", patches=1):
        self.img_dir = img_dir
        self.ref_dir = ref_dir
        self.ycbcr_transform = ycbcr_transform
        self.rgb_transform = rgb_transform
        self.backbone_transform = backbone_transform
        self.refs = refs
        self.df = pd.read_csv(csv_path)
        dfs = [self.df[self.df["image"] == ref] for ref in refs]
        self.df = pd.concat(dfs)
        self.len = len(self.df)
        self.type = type
        self.patches = patches
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_info = self.df.iloc[idx].to_dict()

        # 6 = num of dst levels
        img_info['label'] = (img_info['dst_idx']-1)*5 + img_info['dst_lev']-1
        img_path = f"{self.img_dir}{img_info['dst_type']}/{img_info['image']}.{img_info['dst_type']}.{img_info['dst_lev']}.png"
        

        img_info['pic_path'] = img_path
        img_info['ref_path'] = f"{self.ref_dir}{img_info['image']}.png"
        img_info['metric'] = img_info['dmos']

        ycbcr = Image.open(img_path).convert('YCbCr')
        if self.ycbcr_transform:
            ycbcr = self.ycbcr_transform(ycbcr)

        if self.type == "train":
            return ycbcr, img_info

        rgb = Image.open(img_path)
        if self.rgb_transform:
            rgb_1 = self.rgb_transform(rgb)
        if self.backbone_transform:
            rgb_2 = []
            for _ in range(self.patches):
                rgb_2.append(self.backbone_transform(rgb))
        return ycbcr, rgb_1, torch.stack(rgb_2), img_info

class Koniq10k(Dataset):
    def __init__(self, indeces, img_dir, csv_path, ycbcr_transform=None, rgb_transform=None, backbone_transform=None, type="train", patches=1):
        self.img_dir = img_dir
        self.ycbcr_transform = ycbcr_transform
        self.rgb_transform = rgb_transform
        self.backbone_transform = backbone_transform
        self.df = pd.read_csv(csv_path)

        # if type == 'train':
        #     self.df = self.df[self.df["set"] == "training"]
        # elif type == 'test':
        #     self.df = self.df[self.df["set"] == "test"]
        # elif type == 'val':
        #     self.df = self.df[self.df["set"] == "validation"]
        # elif type == 'test+train':
        #     df1 = self.df[self.df["set"] == "training"]
        #     df2 = self.df[self.df["set"] == "test"]
        #     self.df = pd.concat([df1, df2])
        # elif type == 'val+train':
        #     df1 = self.df[self.df["set"] == "training"]
        #     df2 = self.df[self.df["set"] == "validation"]
        #     self.df = pd.concat([df1, df2])

        self.df = self.df.iloc[indeces]
        self.len = len(self.df)
        self.type = type
        self.patches = patches
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_info = self.df.iloc[idx].to_dict()

        # 6 = num of dst levels
        img_path = f"{self.img_dir}{img_info['image_name']}"
        

        img_info['pic_path'] = img_path
        img_info['metric'] = img_info['MOS']

        ycbcr = Image.open(img_path).convert('YCbCr')
        if self.ycbcr_transform:
            ycbcr = self.ycbcr_transform(ycbcr)

        if self.type == "train" or self.type == 'test+train' or self.type == 'val+train':
            return ycbcr, img_info

        rgb = Image.open(img_path)
        if self.rgb_transform:
            rgb_1 = self.rgb_transform(rgb)
        if self.backbone_transform:
            rgb_2 = []
            for _ in range(self.patches):
                rgb_2.append(self.backbone_transform(rgb))
        return ycbcr, rgb_1, torch.stack(rgb_2), img_info

class LiveITW(Dataset):
    def __init__(self, img_dir, csv_path, ycbcr_transform=None, rgb_transform=None, backbone_transform=None, type="test", patches=1):
        self.img_dir = img_dir
        self.ycbcr_transform = ycbcr_transform
        self.rgb_transform = rgb_transform
        self.backbone_transform = backbone_transform
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df)
        self.type = type
        self.patches = patches
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_info = self.df.iloc[idx].to_dict()

        # 6 = num of dst levels
        img_path = f"{self.img_dir}{img_info['images']}"
        

        img_info['pic_path'] = img_path
        img_info['metric'] = img_info['mos']

        ycbcr = Image.open(img_path).convert('YCbCr')
        if self.ycbcr_transform:
            ycbcr = self.ycbcr_transform(ycbcr)

        if self.type == "train":
            return ycbcr, img_info

        rgb = Image.open(img_path)
        if self.rgb_transform:
            rgb_1 = self.rgb_transform(rgb)
        if self.backbone_transform:
            rgb_2 = []
            for _ in range(self.patches):
                rgb_2.append(self.backbone_transform(rgb))
        return ycbcr, rgb_1, torch.stack(rgb_2), img_info

class Spaq(Dataset):
    def __init__(self, indeces, img_dir, csv_path, ycbcr_transform=None, rgb_transform=None, backbone_transform=None, type="train", patches=1):
        self.img_dir = img_dir
        self.ycbcr_transform = ycbcr_transform
        self.rgb_transform = rgb_transform
        self.backbone_transform = backbone_transform
        self.df = pd.read_csv(csv_path)
        self.df = self.df.iloc[indeces]
        self.len = len(self.df)
        self.type = type
        self.patches = patches
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_info = self.df.iloc[idx].to_dict()

        # 6 = num of dst levels
        img_path = f"{self.img_dir}{img_info['Image name']}"
        

        img_info['pic_path'] = img_path
        img_info['metric'] = img_info['MOS']

        ycbcr = Image.open(img_path).convert('YCbCr')

        if self.ycbcr_transform:
            ycbcr = self.ycbcr_transform(ycbcr)

        if self.type == "train":
            return ycbcr, img_info

        rgb = Image.open(img_path)
        if self.rgb_transform:
            rgb_1 = self.rgb_transform(rgb)
        if self.backbone_transform:
            rgb_2 = []
            for _ in range(self.patches):
                rgb_2.append(self.backbone_transform(rgb))
        return ycbcr, rgb_1, torch.stack(rgb_2), img_info