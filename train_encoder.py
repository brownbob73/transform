#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import torch.utils.data
from torchvision.utils import make_grid, save_image

from encoder import Encoder
from discriminator import Discriminator
from decoder import Decoder
from dataset import FacesDataset
from openface import prepareOpenFace



class Evaluator:
    def __init__(self, encoder, decoder, discriminator, openface):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.openface = openface
        self.upscale_1024 = nn.Upsample(size=(1024,1024))
        self.upscale_768 = nn.Upsample(size=(768,768))
        self.downscale_32x = nn.AvgPool2d(kernel_size=(32,32))
        self.downscale_8x = nn.AvgPool2d(kernel_size=(8,8))

        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.discriminator.parameters():
            p.requires_grad = False
        for p in self.openface.parameters():
            p.requires_grad = False


    def loss(self, input_1024):
        batch_size = input_1024.size()[0]
        input_96 = self.downscale_8x(self.upscale_768(self.downscale_8x(input_1024)))
        input_32 = self.downscale_32x(input_1024)

        input_96_rescaled = (input_96 - input_96.min()) / (input_96.max() - input_96.min())
        input_32_rescaled = (input_32 - input_32.min()) / (input_32.max() - input_32.min())

        input_id_128, _ = self.openface(input_96_rescaled)

        latent_input_features = input_id_128
        latent_512 = self.encoder(latent_input_features).view(-1, 512, 1, 1)
        error_norm = torch.norm(latent_512)

        output_1024 = self.decoder(latent_512)
        output_96 = self.downscale_8x(self.upscale_768(self.downscale_8x(output_1024)))
        output_32 = self.downscale_32x(output_1024)

        output_96_rescaled = (output_96 - output_96.min()) / (output_96.max() - output_96.min())
        output_32_rescaled = (output_32 - output_32.min()) / (output_32.max() - output_32.min())

        output_id_128, _ = self.openface(output_96_rescaled)
        
        error_mse = torch.sum(torch.pow(input_32_rescaled - output_32_rescaled, 2)) / input_32_rescaled.data.nelement()
        similarity = nn.CosineSimilarity()(input_id_128, output_id_128)
#        error_discrim = self.discriminator(output_1024)
        error_discrim = 0

#        loss = torch.mean(-similarity + error_mse + error_norm) #+ error_discrim
        loss = torch.mean(-similarity + error_norm/16.0)
        return loss, (-similarity, error_norm, error_mse, error_discrim), output_1024



def load_decoder(path):
    print('Loading decoder')
    decoder = Decoder()
    decoder.load_state_dict(torch.load(args.decoder))
    return decoder


def get_last_checkpoint(path, head, tail):
    highest_chkpt_index = -1
    highest_chkpt_path = None
    regex = re.compile(head + '(?P<idx>[0-9]+)' + tail)
    with os.scandir(path) as dir_iter:
        for dir_entry in dir_iter:
            if entry.is_file():
                matchobj = regex.match(entry.name)
                if matchobj != None:
                    this_idx = int(matchobj.group('idx'))
                    if this_idx > highest_chkpt_index:
                        highest_chkpt_index = this_idx
                        highest_chkpt_path = entry.path
    return highest_chkpt_index, highest_chkpt_path


def load_encoder(path):
    encoder = Encoder()
    last_checkpoint_idx, last_checkpoint_path = get_last_checkpoint(path, 'encoder-snapshot-', '\.pth$')
    if last_checkpoint_path == None:
        print('No encoder checkpoints found; starting training from scratch')
    else:
        print(f'Loading encoder from checkpoint {last_checkpoint_idx}, path {last_checkpoint_path}')
        encoder.load_state_dict(torch.load(last_checkpoint_path))
    return encoder


def load_discriminator(path):
    discrim = Discriminator()
    discrim.load_state_dict(torch.load(args.discriminator))
    return discrim


def show_grid(input, output):
    images = torch.cat((input.data, output.data), 0)
    images = make_grid(images, padding=10, normalize=True, scale_each=True).cpu()
    image_np = images.numpy().transpose(1, 2, 0)
    plt.imshow(image_np)
    plt.show()


def save_grid(input, output, file):
    images = torch.cat((input.data, output.data), 0)
    images = save_image(images, file, padding=10, normalize=True, scale_each=True)


def train_encoder_batch(input, evaluator, optimiser, i_batch):
    optimiser.zero_grad()
    loss, subloss, output = evaluator.loss(input)
#    print(f'Iter {i_batch}: IDL {subloss[0].data[0]}  NRM {subloss[1].data[0]}  MSE {subloss[2].data[0]}  DIS {subloss[3].data[0]}')
    print(f'Iter {i_batch}: IDL {subloss[0].data[0]}  NRM {subloss[1].data[0]}  MSE {subloss[2].data[0]}')
    loss.backward()
    optimiser.step()
    if i_batch % 10 == 0:
        output_rescaled = (output - torch.min(output))/(torch.max(output)-torch.min(output))
        save_grid(input, output_rescaled, f'chk{i_batch}.jpg')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inference demo')
    parser.add_argument('--images', default='/mnt/data/celeba/nvidia_hq', type=str, metavar='PATH', help='path to image library')
    parser.add_argument('--decoder', default='models/final/decoder-010804.pth', type=str, metavar='PATH', help='path to decoder model (as PyTorch state dict)')
    parser.add_argument('--discriminator', default='models/final/discriminator-010804.pth', type=str, metavar='PATH', help='path to discriminator model (as PyTorch state dict)')
    parser.add_argument('--openface', default='models/final/openface-20180119.pth', type=str, metavar='PATH', help='path to OpenFace one-shot model (as PyTorch state dict)')
    parser.add_argument('--encoder', default='models/inprogress/encoder/', type=str, metavar='PATH', help='path to encoder model working directory')
    parser.add_argument('--seed', default=314159, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    decoder = load_decoder(args.decoder).cuda()
    encoder = load_encoder(args.encoder).cuda()
    discriminator = load_discriminator(args.discriminator).cuda()
    openface = prepareOpenFace(args.openface).eval()

    evaluator = Evaluator(encoder, decoder, discriminator, openface)

    # Freeze the decoder's parameters to save memory
    for p in decoder.parameters():
        p.requires_grad = False
    for p in discriminator.parameters():
        p.requires_grad = False
    for p in openface.parameters():
        p.requires_grad = False

    print(f'Decoder       parameters: {sum((np.prod(p.size()) for p in decoder.parameters()))} total, {sum((np.prod(p.size()) for p in decoder.parameters() if p.requires_grad))} free')
    print(f'Discriminator parameters: {sum((np.prod(p.size()) for p in discriminator.parameters()))} total, {sum((np.prod(p.size()) for p in discriminator.parameters() if p.requires_grad))} free')
    print(f'Encoder       parameters: {sum((np.prod(p.size()) for p in encoder.parameters()))} total, {sum((np.prod(p.size()) for p in encoder.parameters() if p.requires_grad))} free')
    print(f'OpenFace      parameters: {sum((np.prod(p.size()) for p in openface.parameters()))} total, {sum((np.prod(p.size()) for p in openface.parameters() if p.requires_grad))} free')

    data = FacesDataset(args.images)
    dataloader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True, num_workers=4)

    optimiser = torch.optim.Adam(encoder.parameters())
    
    for i_batch, sample_batched in enumerate(dataloader):
        input_data = Variable(sample_batched.cuda(), requires_grad=False)
        train_encoder_batch(input_data, evaluator, optimiser, i_batch)


        # total = 0
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             total += np.prod(obj.size())
        #     except:
        #         pass
        # print(f'Pre-var: {total/1024/1024}')

