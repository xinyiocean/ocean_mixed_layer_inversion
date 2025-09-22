#!/usr/bin/env python

# contains a U-Net module
# [requirement]:
# python packages: pytorch

import torch
import torch.nn as nn
from .net_blocks import *

class Unet(nn.Module):
    def __init__(self, input_channels, output_channels, filters_num):
        super().__init__()
        self.conv_in = Conv_block(input_channels, filters_num[0])
        self.encoder = nn.ModuleList([
            Down_block(input_num, output_num) \
            for input_num, output_num in zip(filters_num[:-1], filters_num[1:])
        ])
        self.decoder = nn.ModuleList([
            Up_block(input_num, output_num) \
            for input_num, output_num in zip(reversed(filters_num[1:]), reversed(filters_num[:-1]))
        ])
        self.conv_out = conv_lay(filters_num[0], output_channels)
    def forward(self, x):
        x = self.conv_in(x)
        encoder_out = []
        # down-sample
        for down_block in self.encoder:
            encoder_out.append(x)
            x = down_block(x)
        # up-sample
        for up_block, eout in zip(self.decoder, reversed(encoder_out)):
            x = up_block(x, eout)
        x = self.conv_out(x)
        return x
