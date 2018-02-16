# -*- coding: utf-8 -*-
"""
Data loader for a directory of face images.
"""

import os
import skimage.io
import torch.utils.data
from torchvision import datasets, transforms, utils


class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.paths = [entry.path for entry in os.scandir(rootdir) if entry.is_file() and not entry.name.startswith('.') and entry.name.endswith('.png')]


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        image_sk = skimage.io.imread(self.paths[idx])
        image = 2*(transforms.ToTensor()(image_sk) - 0.5)
        return image
