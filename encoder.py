# -*- coding: utf-8 -*-
"""
Simple convnet used to encode a 128x128 image to a 1x512 latent vector.
"""

from collections import OrderedDict
import torch
import torch.nn as nn

from modules import *


# Experiment 1: replicate of discriminator network with MSE loss.
# Seed: 1234
# Result: Worked OK, but large memory burden, and blurry around edges,
# probably due to MSE loss.
#
# Input              -          3 x 128 x 128
# Conv 1x1           LReLU     16 x 128 x 128       64
# 
# Conv 3x3           LReLU     16 x 128 x 128     2304
# Conv 3x3           LReLU     32 x 128 x 128     4608
# Downsample         -         32 x  64 x  64
# 
# Conv 3x3           LReLU     32 x  64 x  64     9216
# Conv 3x3           LReLU     64 x  64 x  64    18432
# Downsample         -         64 x  32 x  32
# 
# Conv 3x3           LReLU     64 x  32 x  32    36864
# Conv 3x3           LReLU    128 x  32 x  32    73728
# Downsample         -        128 x  16 x  16
# 
# Conv 3x3           LReLU    128 x  16 x  16   147456
# Conv 3x3           LReLU    256 x  16 x  16   294912
# Downsample         -        256 x   8 x   8
# 
# Conv 3x3           LReLU    256 x   8 x   8   589824
# Conv 3x3           LReLU    512 x   8 x   8  1179648
# Downsample         -        512 x   4 x   4
# 
# Minibatch stddev   -        513 x   4 x   4
# Conv 3x3           LReLU    512 x   4 x   4  2363904
# Conv 4x4           LReLU    512 x   1 x   1  4194304
#                                              -------
#                                              8917776
# 
# 
        # self.layers = nn.Sequential(OrderedDict([
        #     ('conv0.1', nn.Conv2d(3, 16, kernel_size=1, padding=0)),
        #     ('LReLU0.1', nn.LeakyReLU(0.2)),

        #     ('conv1.1', nn.Conv2d(16, 16, kernel_size=3, padding=1)),
        #     ('LReLU1.1', nn.LeakyReLU(0.2)),
        #     ('conv1.2', nn.Conv2d(16, 32, kernel_size=3, padding=1)),
        #     ('LReLU1.2', nn.LeakyReLU(0.2)),
        #     ('dsamp1', nn.AvgPool2d(kernel_size=2, stride=2)),

        #     ('conv2.1', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        #     ('LReLU2.1', nn.LeakyReLU(0.2)),
        #     ('conv2.2', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
        #     ('LReLU2.2', nn.LeakyReLU(0.2)),
        #     ('dsamp2', nn.AvgPool2d(kernel_size=2, stride=2)),

        #     ('conv3.1', nn.Conv2d(64, 64, kernel_size=3, padding=1)),
        #     ('LReLU3.1', nn.LeakyReLU(0.2)),
        #     ('conv3.2', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
        #     ('LReLU3.2', nn.LeakyReLU(0.2)),
        #     ('dsamp3', nn.AvgPool2d(kernel_size=2, stride=2)),

        #     ('conv4.1', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
        #     ('LReLU4.1', nn.LeakyReLU(0.2)),
        #     ('conv4.2', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
        #     ('LReLU4.2', nn.LeakyReLU(0.2)),
        #     ('dsamp4', nn.AvgPool2d(kernel_size=2, stride=2)),

        #     ('conv5.1', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        #     ('LReLU5.1', nn.LeakyReLU(0.2)),
        #     ('conv5.2', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
        #     ('LReLU5.2', nn.LeakyReLU(0.2)),
        #     ('dsamp5', nn.AvgPool2d(kernel_size=2, stride=2)),

        #     ('stdev', StdevAugLayer()),
        #     ('conv6.1', nn.Conv2d(513, 512, kernel_size=3, padding=1)),
        #     ('LReLU6.1', nn.LeakyReLU(0.2)),
        #     ('conv6.2', nn.Conv2d(512, 512, kernel_size=4, stride=4, padding=0)),
        #     ('LReLU6.2', nn.LeakyReLU(0.2))
        # ]))


# Experiment 2: As per experiment 1, but with final tanh layer.
# Seed: 1234
# Result: almost instant mode collapse.


# Experiment 3: 

# 64 for the NIN
# 16*9*16 + 1 = 2305
# 2369-2304=65

# D9b          (None, 16, 1024, 1024)         2369      
# D9bws        (None, 16, 1024, 1024)         2386
# D9a          (None, 32, 1024, 1024)         6994
# D9aws        (None, 32, 1024, 1024)         7027
# D9dn         (None, 32, 512, 512)           7027
# D8b          (None, 32, 512, 512)           16372
# D8bws        (None, 32, 512, 512)           16405
# D8a          (None, 64, 512, 512)           34837
# D8aws        (None, 64, 512, 512)           34902
# D8dn         (None, 64, 256, 256)           34902
# D7b          (None, 64, 256, 256)           72023
# D7bws        (None, 64, 256, 256)           72088
# D7a          (None, 128, 256, 256)          145816
# D7aws        (None, 128, 256, 256)          145945
# D7dn         (None, 128, 128, 128)          145945
# D6b          (None, 128, 128, 128)          293914
# D6bws        (None, 128, 128, 128)          294043
# D6a          (None, 256, 128, 128)          588955
# D6aws        (None, 256, 128, 128)          589212
# D6dn         (None, 256, 64, 64)            589212
# D5b          (None, 256, 64, 64)            1180061
# D5bws        (None, 256, 64, 64)            1180318
# D5a          (None, 512, 64, 64)            2359966
# D5aws        (None, 512, 64, 64)            2360479
# D5dn         (None, 512, 32, 32)            2360479
# D4b          (None, 512, 32, 32)            4721824
# D4bws        (None, 512, 32, 32)            4722337
# D4a          (None, 512, 32, 32)            7081633
# D4aws        (None, 512, 32, 32)            7082146
# D4dn         (None, 512, 16, 16)            7082146
# D3b          (None, 512, 16, 16)            9443491
# D3bws        (None, 512, 16, 16)            9444004
# D3a          (None, 512, 16, 16)            11803300
# D3aws        (None, 512, 16, 16)            11803813
# D3dn         (None, 512, 8, 8)              11803813
# D2b          (None, 512, 8, 8)              14165158
# D2bws        (None, 512, 8, 8)              14165671
# D2a          (None, 512, 8, 8)              16524967
# D2aws        (None, 512, 8, 8)              16525480
# D2dn         (None, 512, 4, 4)              16525480
# Dstat        (None, 513, 4, 4)              16527529  2049  +1e-8    return 
# D1b          (None, 512, 4, 4)              18891433
# D1bws        (None, 512, 4, 4)              18891946
# D1a          (None, 512, 1, 1)              23086250
# D1aws        (None, 512, 1, 1)              23086763


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('D9x', NormConvBlock(3, 16, kernel_size=1, padding=0)),

            ('D9b', NormConvBlock(16, 16, kernel_size=3, padding=1)),
            ('D9a', NormConvBlock(16, 32, kernel_size=3, padding=1)),
            ('D9dn', nn.AvgPool2d(kernel_size=2, stride=2)),

            ('D8b', NormConvBlock(32, 32, kernel_size=3, padding=1)),
            ('D8a', NormConvBlock(32, 64, kernel_size=3, padding=1)),
            ('D8dn', nn.AvgPool2d(kernel_size=2, stride=2)),

            ('D7b', NormConvBlock(64, 64, kernel_size=3, padding=1)),
            ('D7a', NormConvBlock(64, 128, kernel_size=3, padding=1)),
            ('D7dn', nn.AvgPool2d(kernel_size=2, stride=2)),

            ('D6b', NormConvBlock(128, 128, kernel_size=3, padding=1)),
            ('D6a', NormConvBlock(128, 256, kernel_size=3, padding=1)),
            ('D6dn', nn.AvgPool2d(kernel_size=2, stride=2)),

            ('D5b', NormConvBlock(256, 256, kernel_size=3, padding=1)),
            ('D5a', NormConvBlock(256, 512, kernel_size=3, padding=1)),
            ('D5dn', nn.AvgPool2d(kernel_size=2, stride=2)),

            ('D4b', NormConvBlock(512, 512, kernel_size=3, padding=1)),
            ('D4a', NormConvBlock(512, 512, kernel_size=3, padding=1)),
            ('D4dn', nn.AvgPool2d(kernel_size=2, stride=2)),

            ('D3b', NormConvBlock(512, 512, kernel_size=3, padding=1)),
            ('D3a', NormConvBlock(512, 512, kernel_size=3, padding=1)),
            ('D3dn', nn.AvgPool2d(kernel_size=2, stride=2)),

            ('D2b', NormConvBlock(512, 512, kernel_size=3, padding=1)),
            ('D2a', NormConvBlock(512, 512, kernel_size=3, padding=1)),
            ('D2dn', nn.AvgPool2d(kernel_size=2, stride=2)),

            ('Dstat', StdevAug()),
            ('D1b', NormConvBlock(513, 512, kernel_size=3, padding=1)),
            ('D1a', NormConvBlock(512, 512, kernel_size=4, stride=4, padding=0)),

            ('Dscores', NormDenseBlock(512, 1))
        ]))


    def forward(self, x):
        return self.features(x)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128, 32)),
            ('fc1n', nn.LeakyReLU()),
            ('fc2', nn.Linear(32, 32)),
            ('fc2n', nn.LeakyReLU()),
            ('fc3', nn.Linear(32, 32)),
            ('fc3n', nn.LeakyReLU()),
            ('fc4', nn.Linear(32, 32)),
            ('fc4n', nn.LeakyReLU()),
            ('fc5', nn.Linear(32, 32)),
            ('fc5n', nn.LeakyReLU()),
            ('fc6', nn.Linear(32, 512)),
            ('fc6n', nn.LeakyReLU())
            # ('norm', UnitNorm())
        ]))


    def forward(self, x):
        return self.layers(x)



# # Ideas: use openface + high-level image props network (trained)
# # IP network uses savage downsample as first stage, only operates on
# # small image.  Conv NN with 32 outputs.

# Change from MSE -- this seems to be encouraging blurriness in variable regions.

# We train the networks using Adam (Kingma & Ba, 2015) with
# α = 0.001
# β1 = 0
# β2 = 0.99
# eps = 1e−8

# We use a minibatch size 16 for resolutions 4^2–128^2 and gradually then decrease the size according to 256^2→14, 512^2→6, and 1024^2→3 to avoid exceeding the available memory budget