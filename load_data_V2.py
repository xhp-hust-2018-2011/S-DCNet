#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:41:27 2018

@author: xionghaipeng
"""

__author__='xhp'

'''load the dataset'''
#from __future__ import print_function, division
import os
import torch
#import pandas as pd #load csv file
from skimage import io, transform#
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset#, DataLoader#

#from torchvision import transforms, utils#
import glob#use glob.glob to get special flielist
import scipy.io as sio#use to import mat as dic,data is ndarray

import torch
import torch.nn.functional as F


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class myDataset(Dataset):
    """dataset. also can be used for annotation like density map"""

    def __init__(self, img_dir,tar_dir, rgb_dir,transform=None,if_test = False,\
        IF_loadmem=False):
        """
        Args:
            img_dir (string ): Directory with all the images.
            tar_dir (string ): Path to the annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.IF_loadmem = IF_loadmem #whether to load data in memory
        self.IF_loadFinished = False
        self.image_mem = []
        self.target_mem = []

        self.img_dir = img_dir
        self.tar_dir = tar_dir
        self.transform = transform
        
        mat = sio.loadmat(rgb_dir)
        self.rgb = mat['rgbMean'].reshape(1,1,3) #rgbMean is computed after norm to [0-1]
        
        
        img_name = os.path.join(self.img_dir,'*.jpg')
        self.filelist =  glob.glob(img_name)
        self.dataset_len = len(self.filelist)
        
        # for test process, load data is different
        self.if_test = if_test

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        # ------------------------------------
        # 1. see if load from disk or memory
        # ------------------------------------
        if (not self.IF_loadmem) or (not self.IF_loadFinished): 
            img_name =self.filelist[idx]
            image = io.imread(img_name) #load as numpy ndarray
            image = image/255. -self.rgb #to normalization,auto to change dtype
            
            (filepath,tempfilename) = os.path.split(img_name)
            (name,extension) = os.path.splitext(tempfilename)
            
            mat_dir = os.path.join( self.tar_dir, '%s.mat' % (name) )
            mat = sio.loadmat(mat_dir)

            # if need to save in memory
            if self.IF_loadmem:
                self.image_mem.append(image)
                self.target_mem.append(mat)
                # updata if load finished
                if len(self.image_mem) == self.dataset_len:
                    self.IF_loadFinished = True

        else:
            image = self.image_mem[idx]
            mat = self.target_mem[idx]
            #target = mat['target']
        
        # for train may need pre load
        if not self.if_test:
            target = mat['crop_gtdens']
            sample = {'image': image, 'target': target}
            if self.transform:
                sample = self.transform(sample)

            # pad the image
            sample['image'],sample['target'] = get_pad(sample['image'],DIV=64),get_pad(sample['target'],DIV=64)
        else:
            target = mat['all_num']
            sample = {'image': image, 'target': target}
            if self.transform:
                sample = self.transform(sample)
            sample['density_map'] = torch.from_numpy(mat['density_map'])

            # pad the image
            sample['image'],sample['density_map'] = get_pad(sample['image'],DIV=64),get_pad(sample['density_map'],DIV=64)

        return sample
    
    
######################################################################
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'target': torch.from_numpy(target)}


######################################################################
def get_pad(inputs,DIV=64):
    h,w = inputs.size()[-2:]
    ph,pw = (DIV-h%DIV),(DIV-w%DIV)
    # print(ph,pw)

    if (ph!=DIV) or (pw!=DIV):
        tmp_pad = [pw//2,pw-pw//2,ph//2,ph-ph//2]
        # print(tmp_pad)
        inputs = F.pad(inputs,tmp_pad)

    return inputs

if __name__ =='__main__':
    inputs = torch.ones(6,60,730,970);print('ori_input_size:',str(inputs.size()) )
    inputs = get_pad(inputs);print('pad_input_size:',str(inputs.size()) )
