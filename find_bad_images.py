#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import skimage.io
import numpy as np


if __name__ == '__main__':
    with os.scandir('./images/final/256') as it:
        for entry in it:
            if entry.is_file() and not entry.name.startswith('.') and entry.name.endswith('.jpg'):
                image_sk = skimage.io.imread(entry.path)
                if len(image_sk.shape) != 3 or np.prod(image_sk.shape) == 0:
                    print(entry.path)
