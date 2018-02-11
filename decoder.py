#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/ptrblck/prog_gans_pytorch_inference/blob/master/model.py
which in turn was based on the Theano/Lasagne implementation of
Progressive Growing of GANs paper from tkarras:
https://github.com/tkarras/progressive_growing_of_gans
"""

import torch
import torch.nn as nn

from collections import OrderedDict
from modules import *


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1),
            NormConvBlock(256, 256, kernel_size=3, padding=1),
            NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1),
            NormConvBlock(128, 128, kernel_size=3, padding=1),
            NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1),
            NormConvBlock(64, 64, kernel_size=3, padding=1),
            NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1),
            NormConvBlock(32, 32, kernel_size=3, padding=1),
            NormUpscaleConvBlock(32, 16, kernel_size=3, padding=1),
            NormConvBlock(16, 16, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
                        ('norm', PixelNormLayer()),
                        ('conv', nn.Conv2d(16,
                                           3,
                                           kernel_size=1,
                                           padding=0,
                                           bias=False)),
                        ('wscale', WScaleLayer(3))
                    ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x
