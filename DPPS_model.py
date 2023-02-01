#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:23:25 2023

@author: ruby
"""

import torch.nn as nn
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet



def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1,
                                   dilation=dilation, bias=bias)
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    if bias:
        init.constant(layer.bias, 0)
    return layer

# Returns 2D batch normalisation layer
def bn(planes):
    layer = nn.BatchNorm2d(planes)
    # Use mean 0, standard deviation 1 init
    init.constant(layer.weight, 1)
    init.constant(layer.bias, 0)
    return layer


class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
        self.conv_f = conv(96,64,stride = 1)

    def forward(self, x):
        x1 = self.conv_f(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x2 = self.maxpool(x)
        x = self.layer1(x2)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5


class DPPSnet(nn.Module):
    def __init__(self, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.conv5 = conv(512, 256, stride=2, transposed=True)
        self.bn5 = bn(256)
        self.conv6 = conv(256, 128, stride=2, transposed=True)
        self.bn6 = bn(128)
        self.conv7 = conv(128, 64, stride=2, transposed=True)
        self.bn7 = bn(64)
        self.conv8 = conv(64, 64, stride=2, transposed=True)
        self.bn8 = bn(64)
        self.conv9 = conv(64, 32, stride=2, transposed=True)
        self.bn9 = bn(32)
        self.convadd = conv(32, 16, stride=1, transposed=False)
        self.bnadd = bn(16)
        self.conv10 = conv(16, 3,stride=2, kernel_size=5)
        init.constant(self.conv10.weight, 0)  # Zero init
        
        self.conv_d0 = conv(512, 256, stride=2, transposed=True)
        self.bn_d0 = bn(256)
        self.conv_d1 = conv(256, 128, stride=2, transposed=True)
        self.bn_d1 = bn(128)
        self.conv_d2 = conv(128, 64, stride=2, transposed=True)
        self.bn_d2 = bn(64)
        self.conv_d3 = conv(64, 64, stride=2, transposed=True)
        self.bn_d3 = bn(64)
        self.conv_d4 = conv(64, 32, stride=2, transposed=True)
        self.bn_d4 = bn(32)
        self.conv_d5 = conv(32, 16, stride=1, transposed=False)
        self.bn_d5 = bn(16)
        
        self.convd = conv(16, 1,stride=2, kernel_size=5)
        
        init.constant(self.convd.weight, 0)  # Zero init

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.pretrained_net(x)
        x = self.relu(self.bn5(self.conv5(x5)))
        xd0 = self.relu(self.bn_d0(self.conv_d0(x5)))
        x = self.relu(self.bn6(self.conv6(x + x4)))
        xd0 = self.relu(self.bn_d1(self.conv_d1(xd0)))
        xd1 = self.relu(self.bn_d2(self.conv_d2(xd0+x)))
        x = self.relu(self.bn7(self.conv7(x + x3)))
       
        xd1 = self.relu(self.bn_d3(self.conv_d3(xd1+x)))
        x = self.relu(self.bn8(self.conv8(x + x2)))
        
        xd1 = self.relu(self.bn_d4(self.conv_d4(xd1+x)))
        x = self.relu(self.bn9(self.conv9(x + x1)))
        xn2 = self.relu(self.bnadd(self.convadd(x)))
        x_1 = self.conv10(xn2)
        
        xd2 = self.relu(self.bn_d5(self.conv_d5(xd1+x)))
        x_2 = self.convd(xd2)
        
        return x_1, x_2


