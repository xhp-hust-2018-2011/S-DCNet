# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Func1: change density map into count map
# density map: batch size * 1 * w * h
def get_local_count(density_map,psize,pstride):
    IF_gpu = torch.cuda.is_available() # if gpu, return gpu
    IF_ret_gpu = (density_map.device.type == 'cuda')
    psize,pstride = int(psize),int(pstride)
    density_map = density_map.cpu().type(torch.float32)
    conv_kernel = torch.ones(1,1,psize,psize,dtype = torch.float32)
    if IF_gpu:
        density_map,conv_kernel = density_map.cuda(),conv_kernel.cuda()
    
    count_map = F.conv2d(density_map,conv_kernel,stride=pstride)
    if not IF_ret_gpu:
        count_map = count_map.cpu()
    
    return count_map


# Func2: convert count to class (0->c-1)
def Count2Class(count_map,label_indice):
    if isinstance(label_indice,np.ndarray):
        label_indice = torch.from_numpy(label_indice) 
    IF_gpu = torch.cuda.is_available()
    IF_ret_gpu = (count_map.device.type == 'cuda')        
    label_indice  = label_indice.cpu().type(torch.float32)
    cls_num = len(label_indice)+1
    cls_map = torch.zeros(count_map.size()).type(torch.LongTensor) 
    if IF_gpu:
        count_map,label_indice,cls_map = count_map.cuda(),label_indice.cuda(),cls_map.cuda()
    
    for i in range(cls_num-1):
        if IF_gpu:
            cls_map = cls_map + (count_map >= label_indice[i]).cpu().type(torch.LongTensor).cuda()
        else:
            cls_map = cls_map + (count_map >= label_indice[i]).cpu().type(torch.LongTensor)
    if not IF_ret_gpu:
        cls_map = cls_map.cpu() 
    return cls_map


# Func3: convert class (0->c-1) to count number
def Class2Count(pre_cls,label_indice): 
    '''
    # --Input:
    # 1.pre_cls is class label range in [0,1,2,...,C-1]
    # 2.label_indice not include 0 but the other points
    # --Output:
    # 1.count value, the same size as pre_cls
    '''   
    if isinstance(label_indice,np.ndarray):
        label_indice = torch.from_numpy(label_indice)
    label_indice = label_indice.squeeze()
    IF_gpu = torch.cuda.is_available()
    IF_ret_gpu = (pre_cls.device.type == 'cuda')  
    
    # tranform interval to count value map
    label2count = [0.0]
    for (i,item) in enumerate(label_indice):
        if i<label_indice.size()[0]-1:
            tmp_count = (label_indice[i]+label_indice[i+1])/2
        else:
            tmp_count = label_indice[i]
        label2count.append(tmp_count)
    label2count = torch.tensor(label2count)
    label2count = label2count.type(torch.FloatTensor)

    #outputs = outputs.max(dim=1)[1].cpu().data
    ORI_SIZE = pre_cls.size()
    pre_cls = pre_cls.reshape(-1).cpu()
    pre_counts = torch.index_select(label2count,0,pre_cls.cpu().type(torch.LongTensor))
    pre_counts = pre_counts.reshape(ORI_SIZE)

    if IF_ret_gpu:
        pre_counts = pre_counts.cuda()

    return pre_counts


if __name__ == '__main__':
    pre_cls = torch.Tensor([[0,1,2],[3,4,4]])
    label_indice =torch.Tensor([0.5,1,1.5,2]) 

    pre_counts = Class2Count(pre_cls,label_indice)
    print(pre_cls)
    print(label_indice)
    print(pre_counts)

    pre_cls = Count2Class(pre_counts,label_indice)
    print(pre_cls)