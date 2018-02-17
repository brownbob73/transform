# -*- coding: utf-8 -*-
"""
Simple convnet used to encode a 128x128 image to a 1x512 latent vector.
"""

from collections import OrderedDict
import torch
import torch.nn as nn

from modules import *


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(320, 32)),
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
