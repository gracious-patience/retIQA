from typing import Any
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import os
import glob
from PIL import Image



def load_data(args):

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
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.448, 0.483, 0.491],
                std=[0.248, 0.114, 0.106]
            ),
            transforms.Resize((args.img_height, args.img_width)),
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
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.448, 0.483, 0.491],
                std=[0.248, 0.114, 0.106]
            ),
            transforms.Resize((args.img_height, args.img_width)),
        ])
        dataset = DistortedKadid10k(args.data_path, transform=train_transform)
        print(len(dataset))

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader


class DistortedTids2013(Dataset):
    def __init__(self, img_dir, transform=None) -> None:
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len([name for name in os.listdir(self.img_dir)])
    
    def __getitem__(self, idx):
        img_name = os.listdir(self.img_dir)[idx]
        img_path = self.img_dir + "/" + img_name
        image = Image.open(img_path)
        _ , dist, level = img_name.split('_')
        label = (int(dist)-1)*5 + int(level.split('.')[0])-1
        if self.transform:
            image = self.transform(image)
        return image, label
    
class DistortedKadid10k(Dataset):
    def __init__(self, img_dir, transform=None) -> None:
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len([name for name in os.listdir(self.img_dir)])
    
    def __getitem__(self, idx):
        img_name = os.listdir(self.img_dir)[idx]
        img_path = self.img_dir + "/" + img_name
        image = Image.open(img_path).convert('YCbCr')
        _ , dist, level = img_name.split('_')
        label = (int(dist)-1)*5 + int(level.split('.')[0])-1
        if self.transform:
            image = self.transform(image)
        return image, label

class DistortedKadis700k(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):

        
        return len(glob.glob(os.path.join(self.img_dir,  '*.bmp')))
    
    def __getitem__(self, idx):
        img_path = glob.glob(os.path.join(self.img_dir,  '*.bmp'))[idx]
        img_name = img_path.split('/')[-1]
        image = Image.open(img_path).convert('YCbCr')
        _ , dist, level = img_name.split('_')

        # 5 = number of distorsion levels
        label = (int(dist)-1)*5 + int(level.split('.')[0])-1
        if self.transform:
            image = self.transform(image)
        return image, label