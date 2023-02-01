#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:56:47 2023

@author: ruby
"""
import torch
import torch.utils.data as data_utils
import scipy.io as sio
from PIL import Image
from torchvision.transforms import ToTensor

def default_loader(path):
    return Image.open(path) 

class DPPS_Dataset(data_utils.Dataset):
    def __init__(self, dataset, transform=None, target_transform=None, loader=default_loader):
        '''
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
            
        '''
 
        self.imgs = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        label_x, label_h, label_n = self.imgs[index]
        imgs = torch.empty(0,128,128)
        for i in range(96):
            img_temp = self.loader(label_x+str(i+1)+'.png')
            img_temp = ToTensor()(img_temp)
            imgs = torch.cat((imgs, img_temp), 0)
        gt1 = sio.loadmat(label_h)['position_map'].astype(float)
        gt2 = sio.loadmat(label_n)['normal_map'].astype(float).transpose(2,0,1)*2-1.0

        #f = interpolate.interp2d(x, y, gt, kind='cubic')
        #gt2_ = f(xnew, ynew)
        return imgs,gt1,gt2

    def __len__(self):
        return len(self.imgs)