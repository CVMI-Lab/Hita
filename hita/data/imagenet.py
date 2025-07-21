#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""
import torch, pdb
import numpy as np
import re, pdb, cv2
from PIL import Image
import os.path as osp
import nori2 as nori
from einops import repeat, rearrange
from torch.utils.data import Dataset

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

        return img, target

    def __len__(self):

        return len(self.samples)
