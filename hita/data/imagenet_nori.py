#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""
import numpy as np
import re, pdb, cv2
from PIL import Image
import os.path as osp
import torch, io, pdb
import nori2 as nori
from megfile import smart_open
from .register import set_aws_a
from torchvision import transforms
from einops import repeat, rearrange
from torch.utils.data import Dataset
from .augmentation import random_crop_arr, center_crop_arr

class ImageNet(Dataset):
    """ImageNet dataset."""
    def __init__(self, anno_file = None, samples=None, transform=None):

        self.anno_file = anno_file
        self.transform = transform
        self.nori_fetcher = None

        self.samples = None
        if (anno_file is not None):
            assert osp.exists(anno_file)
            self._decode_nori_list()
        else:
            assert samples is not None
            self.samples = samples
    
    def _decode_nori_list(self):

        self.samples = []
        with open(self.anno_file, "r") as f:
            for line in f:
                nori_id, target, _ = line.strip().split()
                self.samples.append((nori_id, int(target)))

    def _check_nori_fetcher(self):

        """Lazy initialize nori fetcher. In this way, `NoriDataset` can be pickled and used
            in multiprocessing.
        """
        if self.nori_fetcher is None:
            self.nori_fetcher = nori.Fetcher()

    def __getitem__(self, index):

        # Load the image
        self._check_nori_fetcher()
        nori_id, target = self.samples[index]
        img_bytes = self.nori_fetcher.get(nori_id)
        sample = cv2.imdecode(np.fromstring(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        color_converted = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(color_converted)
        if self.transform:
            img = self.transform(img)

        return img, target, nori_id

    def __len__(self):

        return len(self.samples)

class ImageNetDataset(ImageNet):

    def __init__(self, anno_file, image_size, is_train = False):

        super().__init__(anno_file,)
        if is_train:
            # crop_size = int(image_size * crop_range)
            crop_size = image_size
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, crop_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                # transforms.TenCrop(image_size), # this is a tuple of PIL Images
                # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        else:
            crop_size = image_size 
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
    
    def __getitem__(self, idx):

        image, target, nori_id = super().__getitem__(idx)
        return image, target, nori_id

class ImageNetNori(Dataset):

    def __init__(self, anno_file=None, samples=None, transform=None):
        
        set_aws_a()
        self.anno_file = anno_file
        self.transform = transform
        self.samples = None
        if (self.anno_file is not None):
            assert osp.exists(self.anno_file)
            self._decode_nori_list()

        else:
            assert samples is not None
            self.samples = samples
        np.random.shuffle(self.samples)
    
    def __getitem__(self, index):

        nori_path, target = self.samples[index]
        with smart_open(nori_path, mode='rb') as f:
            f.seek(0)
            bytes_data = f.read()
            image = Image.open(io.BytesIO(bytes_data), "r").convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

    def _decode_nori_list(self):

        self.samples = []
        with open(self.anno_file, "r") as f:
            for line in f:
                nori_path, target = line.strip('\n').split()
                self.samples.append((nori_path, int(target)))
        
    def __len__(self):

        return len(self.samples)

class ImageNetNoriDataset(ImageNetNori):

    def __init__(self, anno_file, image_size, crop_range=1., is_train = False):

        super().__init__(anno_file)
        if is_train:
            # crop_size = int(image_size * crop_range)
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                # transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        else:
            crop_size = image_size 
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
    
    def __getitem__(self, idx):

        image, target = super().__getitem__(idx)
        return image, target