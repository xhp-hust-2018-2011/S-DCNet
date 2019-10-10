# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 20:06:33 2018

@author: poppinace
"""

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import  DataLoader
import torch.nn.functional as F
#from torchvision import models

import os
import numpy as np
from time import time

import math
import pandas as pd
import csv

from IOtools import txt_write 


        
def test_phase(opt,net,testloader,log_save_path=None):
    with torch.no_grad():
        net.eval()
        start = time()
        avg_frame_rate = 0
        mae = 0.0
        rmse = 0.0
        me = 0.0

        for j, data in enumerate(testloader):
            inputs , labels = data['image'], data['target']
            inputs,labels = inputs.type(torch.float32),labels.unsqueeze(1).type(torch.float32)
            inputs, labels = inputs.cuda(), labels.cuda()
            

            features = net(inputs)
            div_res = net.resample(features)
            merge_res = net.parse_merge(div_res)
            outputs = merge_res['div'+str(net.args['div_times'])]
            del merge_res
         
            pre =  (outputs).sum()
            gt = labels.sum()
                                             
            mae += abs(pre-gt)
            rmse += (pre-gt)*(pre-gt)
            me += (pre-gt)
            end = time()
            running_frame_rate = opt['test_batch_size'] * float( 1 / (end - start))
            avg_frame_rate = (avg_frame_rate*j + running_frame_rate)/(j+1)
            if j % 1 == 0:    # print every 2000 mini-batches
                print('Test:[%5d/%5d] pre: %.3f gt:%.3f err:%.3f frame: %.2fHz/%.2fHz' %
                        ( j + 1,len(testloader), pre, gt,pre-gt,
                        running_frame_rate,avg_frame_rate) )
                start = time()
        
        log_str =  '%10s\t %8s\t &%8s\t &%8s\t\\\\' % (' ','mae','rmse','me')+'\n'
        log_str += '%-10s\t %8.3f\t %8.3f\t %8.3f\t' % ( 'test',mae/(j+1),math.sqrt(rmse/(j+1)),me/(j+1) ) + '\n'
            
        if log_save_path:
            txt_write(log_save_path,log_str,mode='w')
        
    im_num = len(testloader)

    return mae/(im_num), math.sqrt(rmse/(im_num)),  me/(im_num)





    






 



