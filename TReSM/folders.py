import pandas as pd
import torch.utils.data as data
from torch import cat
from torch import tensor
import torchvision.transforms as T
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv

def str_2_float_list(pseudolist):
    intermediate = pseudolist.strip('][').split(', ')
    return list(map(float, intermediate))
def str_2_str_list(pseudolist):
    intermediate = pseudolist.strip('][').split(', ')
    return list(map(str, intermediate))

spaq_meta_headers = [
    "Brightness_x",
    "Colorfulness",
    "Contrast",
    "Noisiness",
    "Sharpness",
    "Focal length",
    "F-number",
    "Exposure time",
    "ISO",
    "Brightness_y",
    "Flash",
    "Animal",
    "Cityscape",
    "Human",
    "Indoor scene",
    "Landscape",
    "Night scene",
    "Plant",
    "Still-life",
    "Others"
]


class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)

        orgs = dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']


        refname.sort()
        sample = []

        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[0][item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/clive_retr_aug_{seed}.csv")
        df = df.iloc[index]
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                # [1:] slices because 0 item in train is the same as original
                if istrain:
                    sample.append((f"{root}/Images/{row['image']}" , str_2_str_list(row['neighbours'])[1:k+1] ,str_2_float_list(row['neighbours_labels'])[1:] , row['mos']  ))
                # no original pic in test 
                else:
                    sample.append((f"{root}/Images/{row['image']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['mos']  ))

        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('LIVE-itW')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length


class CSIQFolder(data.Dataset):

    def __init__(self, root, seed,  index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/csiq_retr_aug_{seed}.csv")
        ref_names = df["image"].unique()
        dfs = [df[df["image"] == ref_names[ref]] for ref in index]
        df = pd.concat(dfs)
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                # [1:] slices because 0 item in train is the same as original
                if istrain:
                    sample.append((f"{root}/dst_imgs_all/{row['image']}.{row['dst_type']}.{row['dst_lev']}.png" , str_2_str_list(row['neighbours'])[1:k+1] , str_2_float_list(row['neighbours_labels'])[1:], row['dmos'] ))
                # no original pic in test 
                else:
                    sample.append((f"{root}/dst_imgs_all/{row['image']}.{row['dst_type']}.{row['dst_lev']}.png" , str_2_str_list(row['neighbours'])[:k] , str_2_float_list(row['neighbours_labels'])[:], row['dmos'] ))
        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            # [1:-1] slices because I saved pathes like this: "'path'"
            # so, broadcasting it to str returns 'path'
            
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('csiq')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length

class SlyCSIQFolder(data.Dataset):

    def __init__(self, root, seed,  index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/csiq_sly_retr_aug_{seed}.csv")
        ref_names = df["image"].unique()
        dfs = [df[df["image"] == ref_names[ref]] for ref in index]
        df = pd.concat(dfs)
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                # [1:] slices because 0 item in train is the same as original
                if istrain:
                    sample.append((f"{root}/dst_imgs_all/{row['image']}.{row['dst_type']}.{row['dst_lev']}.png" , str_2_str_list(row['neighbours'])[:k] , str_2_float_list(row['neighbours_labels'])[:], row['dmos'] ))
                # no original pic in test 
                else:
                    sample.append((f"{root}/dst_imgs_all/{row['image']}.{row['dst_type']}.{row['dst_lev']}.png" , str_2_str_list(row['neighbours'])[:k] , str_2_float_list(row['neighbours_labels'])[:], row['dmos'] ))
        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            # [1:-1] slices because I saved pathes like this: "'path'"
            # so, broadcasting it to str returns 'path'
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('csiq')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length

class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/koniq_retr_aug_{seed}.csv")
        df = df.iloc[index]
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/512x384/{row['image_name']}" , str_2_str_list(row['neighbours'])[1:k+1] ,str_2_float_list(row['neighbours_labels'])[1:] , row['MOS']  ))
                else:
                    sample.append((f"{root}/512x384/{row['image_name']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['MOS']  ))

        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('koniq10k')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length
    
class Koniq_10kPartialFolder(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k, retrieve_size):
        df = pd.read_csv(f"{root}/koniq_{retrieve_size}_retr_aug_{seed}.csv")
        df = df.iloc[index]
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/512x384/{row['image_name']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['MOS']  ))
                else:
                    sample.append((f"{root}/512x384/{row['image_name']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['MOS']  ))

        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('koniq10k')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length

class Koniq_10kCrossFolder(data.Dataset):
    # original = first picture is loaded from root
    # neighbours = k others -> loaded from cross root
    def __init__(self, root, cross_root, cross_dataset, seed, index, transform, patch_num, istrain, k, delimeter):
        df = pd.read_csv(f"{root}/koniq_cross_{cross_dataset}_retr_aug_{seed}.csv")
        df = df.iloc[index]
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/512x384/{row['image_name']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['MOS']  ))
                else:
                    sample.append((f"{root}/512x384/{row['image_name']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['MOS']  ))

        self.samples = sample
        self.transform = transform
        self.root = root
        self.cross_root = cross_root
        self.cross_dataset = cross_dataset
        self.delimeter = delimeter
        if cross_dataset == 'spaq':
            self.df = pd.read_csv(f"{cross_root}/spaq_info.csv")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.cross_root}{neighbour_path.split(self.delimeter)[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
            if self.cross_dataset == 'spaq':
                # metainfo
                neighbour_name = neighbour_path.split('/')[-1][:-1]
                neighbour_stats = self.df.loc[self.df['Image name'] == neighbour_name]
                metas.append(
                    [
                        list(neighbour_stats[header])[0] for header in spaq_meta_headers
                    ]
                )
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length
    
class BigKoniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, '1024x768', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length
    
class SpaqFolder(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/spaq_retr_aug_{seed}.csv")
        df = df.iloc[index]
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/TestImage/{row['Image name']}" , str_2_str_list(row['neighbours'])[1:k+1] ,str_2_float_list(row['neighbours_labels'])[1:] , row['MOS']  ))
                else:
                    sample.append((f"{root}/TestImage/{row['Image name']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['MOS']  ))

        self.samples = sample
        self.transform = transform
        self.root = root
        self.df = pd.read_csv(f"{root}/spaq_info.csv")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [], []
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            # pics
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('spaq')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
            # metainfo
            neighbour_name = neighbour_path.split('/')[-1][:-1]
            neighbour_stats = self.df.loc[self.df['Image name'] == neighbour_name]
            metas.append(
                [
                    list(neighbour_stats[header])[0] for header in spaq_meta_headers
                ]
            )
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length
    
class SpaqCrossFolder(data.Dataset):

    def __init__(self, root: str, cross_root: str, cross_dataset: str, seed: int, index, transform, patch_num: int, istrain: bool, k: int, delimeter: str):
        df = pd.read_csv(f"{root}/spaq_cross_{cross_dataset}_retr_aug_{seed}.csv")
        df = df.iloc[index]
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/TestImage/{row['Image name']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['MOS']  ))
                else:
                    sample.append((f"{root}/TestImage/{row['Image name']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['MOS']  ))

        self.samples = sample
        self.transform = transform
        self.root = root
        self.cross_root = cross_root
        self.delimeter = delimeter

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [], []
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.cross_root}{neighbour_path.split(self.delimeter)[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length
    
class BiqFolder(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/biq_retr_aug_{seed}.csv")
        df = df.iloc[index]
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/Images/{row['Image Name']}" , str_2_str_list(row['neighbours'])[1:k+1] ,str_2_float_list(row['neighbours_labels'])[1:] , row['MOS']  ))
                else:
                    sample.append((f"{root}/Images/{row['Image Name']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['MOS']  ))

        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('BIQ2021')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length


class FBLIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'labels_image.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['name'].split('/')[1])
                mos = np.array(float(row['mos'])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'FLIVE', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length
     
        

class TID2013Folder(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/tid2013_retr_aug_{seed}.csv")
        ref_names = df["ref"].unique()
        dfs = [df[df["ref"] == ref_names[ref]] for ref in index]
        df = pd.concat(dfs)
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/distorted_images/{row['image']}" , str_2_str_list(row['neighbours'])[1:k+1] ,str_2_float_list(row['neighbours_labels'])[1:] , row['mos']  ))
                else:
                    sample.append((f"{root}/distorted_images/{row['image']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['mos']  ))
        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neighbours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('tid2013')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length

class SlyTID2013Folder(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/tid2013_sly_retr_aug_{seed}.csv")
        ref_names = df["ref"].unique()
        dfs = [df[df["ref"] == ref_names[ref]] for ref in index]
        df = pd.concat(dfs)
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/distorted_images/{row['image']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['mos']  ))
                else:
                    sample.append((f"{root}/distorted_images/{row['image']}" , str_2_str_list(row['neighbours'])[:k] ,str_2_float_list(row['neighbours_labels'])[:] , row['mos']  ))
        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neighbours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('tid2013')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length

class Kadid10k(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/kadid10k_retr_aug_{seed}.csv")
        ref_names = df["reference"].unique()[:-1]
        dfs = [df[df["reference"] == ref_names[ref]] for ref in index]
        df = pd.concat(dfs)
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/distorted_images/{row['image']}" , str_2_str_list(row['neighbours'])[1:k+1] , str_2_float_list(row['neighbours_labels'])[1:] , row['dmos'] ))
                else:
                    sample.append((f"{root}/distorted_images/{row['image']}" , str_2_str_list(row['neighbours'])[:k] , str_2_float_list(row['neighbours_labels'])[:] , row['dmos'] ))
        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]

        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('kadid10k')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length
    
class SlyKadid10k(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/kadid10k_sly_retr_aug_{seed}.csv")
        ref_names = df["reference"].unique()[:-1]
        dfs = [df[df["reference"] == ref_names[ref]] for ref in index]
        df = pd.concat(dfs)
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/distorted_images/{row['image']}" , str_2_str_list(row['neighbours'])[:k] , str_2_float_list(row['neighbours_labels'])[:] , row['dmos'] ))
                else:
                    sample.append((f"{root}/distorted_images/{row['image']}" , str_2_str_list(row['neighbours'])[:k] , str_2_float_list(row['neighbours_labels'])[:] , row['dmos'] ))
        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]

        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('kadid10k')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length

class PipalFolder(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/pipal_retr_aug_{seed}.csv")
        ref_names = df["hq_name"].unique()
        dfs = [df[df["hq_name"] == ref_names[ref]] for ref in index]
        df = pd.concat(dfs)
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/Train_Dist/{row['dist_name']}" , str_2_str_list(row['neighbours'])[1:k+1] , str_2_float_list(row['neighbours_labels'])[1:], row['elo_score'] ))
                else:
                    sample.append((f"{root}/Train_Dist/{row['dist_name']}" , str_2_str_list(row['neighbours'])[:k] , str_2_float_list(row['neighbours_labels'])[:], row['elo_score'] ))

        self.samples = sample
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('train')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length
    
class SlyPipalFolder(data.Dataset):

    def __init__(self, root, seed, index, transform, patch_num, istrain, k):
        df = pd.read_csv(f"{root}/pipal_sly_retr_aug_{seed}.csv")
        ref_names = df["hq_name"].unique()
        dfs = [df[df["hq_name"] == ref_names[ref]] for ref in index]
        df = pd.concat(dfs)
        sample = []

        for _, row in df.iterrows():
            for _ in range(patch_num):
                if istrain:
                    sample.append((f"{root}/Train_Dist/{row['dist_name']}" , str_2_str_list(row['neighbours'])[:k] , str_2_float_list(row['neighbours_labels'])[:], row['elo_score'] ))
                else:
                    sample.append((f"{root}/Train_Dist/{row['dist_name']}" , str_2_str_list(row['neighbours'])[:k] , str_2_float_list(row['neighbours_labels'])[:], row['elo_score'] ))

        self.samples = sample
        self.transform = transform
        self.root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, neighbours, neighbours_target, target = self.samples[index]
        samples, targets, metas = [], [],[]
        
        # main pic
        sample = pil_loader(path)
        sample = self.transform(sample)
        samples.append(sample)
        targets += [target]
        # pics neibours
        for neighbour_path in neighbours:
            sample_neighbour = pil_loader(f"{self.root}{neighbour_path.split('train')[1][:-1]}")
            sample_neighbour = self.transform(sample_neighbour)
            samples.append(sample_neighbour)
        targets += neighbours_target
        return cat(samples, dim=0), tensor(targets), tensor(metas)

    def __len__(self):
        length = len(self.samples)
        return length

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
