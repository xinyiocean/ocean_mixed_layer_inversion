#!/usr/bin/env python

# contains neural network modules
# [requirement]:
# python packages: pytorch

import torch
import torch.nn as nn

def conv_lay(input_num, output_num, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(input_num, output_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

class Conv_block(nn.Module):
    def __init__(self, input_num, output_num, act_lay=nn.ReLU, norm_lay=nn.BatchNorm2d):
        super().__init__()
        # 2 conv layers
        self.convs = nn.Sequential(
            conv_lay(input_num, output_num),
            norm_lay(output_num),
            act_lay(inplace=False),
            conv_lay(output_num, output_num),
            norm_lay(output_num),
            act_lay(inplace=False)
        )
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        x = self.convs(x)
        x = self.drop(x)
        return x

class Down_block(nn.Module):
    def __init__(self, input_num, output_num, kernel_size=2, stride=2):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.convs = Conv_block(input_num, output_num)
    def forward(self, x):
        x = self.down(x)
        x = self.convs(x)
        return x
    
class Up_block(nn.Module):
    def __init__(self, input_num, output_num, kernel_size=2, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_num, input_num, kernel_size=kernel_size, stride=stride)
        self.convs = Conv_block(input_num+output_num, output_num)
    def forward(self, x, eout):
        x = self.up(x)
        x = self.convs(torch.cat([x, eout], dim=1))
        return x

