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
        self.downscale_4x = nn.AvgPool2d(kernel_size=(4,4))
        self.downscale_3x = nn.AvgPool2d(kernel_size=(3,3))

        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.discriminator.parameters():
            p.requires_grad = False
        # for p in self.openface.parameters():
        #     p.requires_grad = False


    def loss(self, input_1024):
        batch_size = input_1024.size()[0]
        input_96 = self.downscale_8x(self.upscale_768(self.downscale_8x(input_1024)))

        input_96_min = input_96.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        input_96_max = input_96.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        input_96_rescaled = (input_96 - input_96_min) / (input_96_max - input_96_min)
        input_32_rescaled = self.downscale_3x(input_96_rescaled)

        input_id_128 = self.openface(input_96_rescaled)[0]                  # B x 128
        
        input_8_rescaled = (self.downscale_4x(input_32_rescaled) - 0.5)*2   # B x 3 x 8 x 8, [-1,1]
        input_8_rescaled_linear = input_8_rescaled.view(-1, 192)

        latent_input_features = torch.cat([input_id_128, input_8_rescaled_linear], 1)
        latent_512 = self.encoder(latent_input_features).view(-1, 512, 1, 1)
        latent_norm = torch.sum(torch.pow(latent_512, 2).squeeze(2).squeeze(2), dim=1)

        output_1024 = self.decoder(latent_512)
        output_96 = self.downscale_8x(self.upscale_768(self.downscale_8x(output_1024)))

        output_96_min = output_96.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        output_96_max = output_96.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        output_96_rescaled = (output_96 - output_96_min) / (output_96_max - output_96_min)
        output_32_rescaled = self.downscale_3x(output_96_rescaled)

        output_id_128 = self.openface(output_96_rescaled)[0]
        
        error_sse = torch.sum(torch.pow(input_32_rescaled - output_32_rescaled, 2), dim=3).sum(dim=2).sum(dim=1)
#        error_discrim = self.discriminator(output_1024)
        error_discrim = 0

        # similarity = torch.sum(input_id_128 * output_id_128, dim=1) / torch.norm(input_id_128, dim=1) / torch.norm(output_id_128, dim=1)
        similarity = nn.CosineSimilarity()(input_id_128, output_id_128)

        loss_sse = error_sse / 100
        loss_norm = torch.abs(latent_norm - 512) / 82
        loss_similarity = similarity / -0.11
        loss_discrim = 0

        #loss = torch.mean(loss_sse + loss_norm)
        loss = torch.mean(loss_similarity)
        return loss, (loss_similarity, loss_norm, loss_sse, loss_discrim), output_1024



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
    save_image(images, file, padding=10, normalize=True, scale_each=True)


def train_encoder_batch(input, evaluator, optimiser, i_batch, i_epoch):
    optimiser.zero_grad()
    loss, subloss, output = evaluator.loss(input)
#    print(f'Iter {i_batch}: IDL {subloss[0].data[0]}  NRM {subloss[1].data[0]}  MSE {subloss[2].data[0]}  DIS {subloss[3].data[0]}')
    print(f'Epoch {i_epoch} Iter {i_batch}: IDL {subloss[0].data[0]}  NRM {subloss[1].data[0]}  SSE {subloss[2].data[0]}')
    loss.backward()
    optimiser.step()
    if i_batch % 10 == 0:
        output_rescaled = (output - torch.min(output))/(torch.max(output)-torch.min(output))
        save_grid(input, output_rescaled, f'chk{i_epoch}_{i_batch}.jpg')



def test_decoder_norm(decoder):
    # Conclusions from testing:
    #  1) Decoder is extremely resistant to norm of latent vector.  Expected values from 1 to 2048 produce almost identical results.
    #  2) Decoder is fairly reisistant to mean of latent vector.  Expected values from -0.1 to 0.1 produce little change.
    #  3) No apparent pattern in LVs that produce poor-quality results.  Assume just under-sampled in the face space.
    #  4) Forcing LVs to lie on the surface of the 512D hypersphere shows no improvement.  (expected from 1)
    #  5) As suggested by 1, the decoder is effectively projecting the LV to the surface of the hypersphere before calculation;
    #     it is almost as if it uses only vector directions.

    # In retrospect the above behaviour is obvious, considering the exclusive use of ReLUs in the generator, and the final
    # normalization stages (PixelNorm, then later image normalization).

    for norm in [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        torch.manual_seed(314159)
        ytot_set = False
        for row in range(4):
            yrow_set = False
            for col in range(6):
                x = Variable(torch.randn(1, 512, 1, 1).cuda(), volatile=True)
                x = x / 512.0 * norm
                print(f'{row+1}, {col+1}, {torch.mean(x).data[0]:.5}, {torch.min(x).data[0]:.5}, {torch.max(x).data[0]:.5}, {torch.sum(torch.pow(x, 2)).data[0]:.5}')
                y = decoder(x)
                y = (y - torch.min(y))/(torch.max(y)-torch.min(y))
                y = torch.nn.AvgPool2d(4)(y).squeeze().cpu()

                if yrow_set == False:
                    yrow_set = True
                    yrow = y
                else:
                    yrow = torch.cat((yrow, y), 2)
                del y
            if ytot_set == False:
                ytot_set = True
                ytot = yrow
            else:
                ytot = torch.cat((ytot, yrow), 1)

        save_image(ytot.data, f'test_{norm}.jpg', padding=10, normalize=False, scale_each=False)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='data/celeba/nvidia_hq', type=str, metavar='PATH', help='path to image library')
    parser.add_argument('--decoder', default='models/final/decoder-010804.pth', type=str, metavar='PATH', help='path to decoder model (as PyTorch state dict)')
    parser.add_argument('--discriminator', default='models/final/discriminator-010804.pth', type=str, metavar='PATH', help='path to discriminator model (as PyTorch state dict)')
    parser.add_argument('--openface', default='models/final/openface-20180119.pth', type=str, metavar='PATH', help='path to OpenFace one-shot model (as PyTorch state dict)')
    parser.add_argument('--encoder', default='models/inprogress/encoder/', type=str, metavar='PATH', help='path to encoder model working directory')
    parser.add_argument('--seed', default=314159, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    decoder = load_decoder(args.decoder).cuda()
    test_decoder_norm(decoder)
    encoder = load_encoder(args.encoder).cuda()
    discriminator = load_discriminator(args.discriminator).cuda()
    openface = prepareOpenFace(args.openface, useCuda=True).eval()

    evaluator = Evaluator(encoder, decoder, discriminator, openface)

    # Freeze the decoder's parameters to save memory
    for p in decoder.parameters():
        p.requires_grad = False
    for p in discriminator.parameters():
        p.requires_grad = False
    # for p in openface.parameters():
    #     p.requires_grad = False

    print(f'Decoder       parameters: {sum((np.prod(p.size()) for p in decoder.parameters()))} total, {sum((np.prod(p.size()) for p in decoder.parameters() if p.requires_grad))} free')
    print(f'Discriminator parameters: {sum((np.prod(p.size()) for p in discriminator.parameters()))} total, {sum((np.prod(p.size()) for p in discriminator.parameters() if p.requires_grad))} free')
    print(f'Encoder       parameters: {sum((np.prod(p.size()) for p in encoder.parameters()))} total, {sum((np.prod(p.size()) for p in encoder.parameters() if p.requires_grad))} free')
    print(f'OpenFace      parameters: {sum((np.prod(p.size()) for p in openface.parameters()))} total, {sum((np.prod(p.size()) for p in openface.parameters() if p.requires_grad))} free')

    data = FacesDataset(args.images)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=1)

    optimiser = torch.optim.Adam(encoder.parameters())
    
    for i_epoch in range(100):
        for i_batch, sample_batched in enumerate(dataloader):
            input_data = Variable(sample_batched.cuda(), requires_grad=False)
            train_encoder_batch(input_data, evaluator, optimiser, i_batch, i_epoch)


        # total = 0
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             total += np.prod(obj.size())
        #     except:
        #         pass
        # print(f'Pre-var: {total/1024/1024}')

