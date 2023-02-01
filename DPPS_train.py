#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:29:07 2023

@author: ruby
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


import io
#import requests


import torch.utils.data as data_utils
import os
#from datasets import pms_transforms
#from datasets import util

from DPPS_model import DPPSnet, FeatureResNet
from DPPS_Dataset import DPPS_Dataset

from datetime import datetime
dataString = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')

fnet = FeatureResNet()
DPPS = DPPSnet(fnet)
DPPS = DPPS.cuda()


EPOCH = 100              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 12
print('BATCH_SIZE = ',BATCH_SIZE)
LR = 0.00005              # learning rate
#root = './gdrive_northwestern/My Drive/dl_encoder/data/orig/orig'
NUM_WORKERS = 0

optimizer = torch.optim.Adam(DPPSnet.parameters(), lr=LR)   # optimize all cnn parameters
#optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9)   # optimize all cnn parameters
loss_func = nn.MSELoss()

file1 = open('train_set.txt', 'r')
Lines = file1.readlines()
train_set = []
for line in Lines:
    new_item = line[:-1].split(',')
    train_set.append(new_item)
    
file1 = open('test_set.txt', 'r')
Lines = file1.readlines()
test_set = []
for line in Lines:
    new_item = line[:-1].split(',')
    test_set.append(new_item)


train_data=DPPS_Dataset(dataset=train_set)
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

test_data=DPPS_Dataset(dataset=test_set)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=1)


Train_root = '/Train_'+dataString[4:10]+'/'
os.mkdir(Train_root)
model_root = Train_root+'model/'
log_root = Train_root+'log/'
os.mkdir(model_root)
os.mkdir(log_root)

fileOut=open(log_root+'log'+dataString,'a')
fileOut.write(dataString+'\n'+'Epoch:   Step:    Loss:        Val_Accu :\n')
fileOut.close()
fileOut2 = open(log_root+'validation'+dataString, 'a')
fileOut2.write(dataString+'\n'+'Epoch:    loss:')
fileOut2.close()

for epoch in range(1,100):
    DPPS.train()
    for step, (img,gt1,gt2) in enumerate(train_loader):   
        img = Variable(img).cuda()
        gt1=gt1.unsqueeze(1).float()
        gt1 = Variable(gt1).cuda()
        gt2=gt2.float()
        gt2 = Variable(gt2).cuda()
        output = DPPS(img)   
        loss1 = loss_func(output[0], gt2)
        loss2 = loss_func(output[1], gt1)
        loss = loss1+loss2
        optimizer.zero_grad()          
        loss.backward()                 
        optimizer.step()  
        if step%50 == 0:
            print(epoch,  step, loss.data.item(),loss1.data.item(),loss2.data.item())
        fileOut=open(log_root+'log'+dataString,'a')
        fileOut.write(str(epoch)+'   '+str(step)+'   '+str(loss.data.item())+'   '+str(loss1.data.item())+'   '+str(loss2.data.item())+'\n')
        fileOut.close()
    if epoch%10 == 9:
        PATH = model_root + 'param_n_3_' + str(epoch) + '_' + str(step)
        torch.save(DPPS.state_dict(), PATH)
        print('finished saving checkpoints')
     
    LOSS_VALIDATION = 0
    DPPS.eval()
    with torch.no_grad():
        for step, (img,gt1,gt2) in enumerate(test_loader):

            img = Variable(img).cuda()
            gt1=gt1.unsqueeze(1).float()
            gt1 = Variable(gt1).cuda()
            gt2=gt2.float()# batch x
            gt2 = Variable(gt2).cuda()
            output = DPPS(img) 
            LOSS_VALIDATION += loss_func(output[1], gt1)+loss_func(output[0], gt2)
        LOSS_VALIDATION = LOSS_VALIDATION/step
        fileOut2 = open(log_root+'validation'+dataString, 'a')
        fileOut2.write(str(epoch)+'   '+str(step)+'   '+str(LOSS_VALIDATION.data.item())+'\n')
        fileOut2.close()
        print('validation error epoch  '+str(epoch)+':    '+str(LOSS_VALIDATION)+'\n'+str(step))



