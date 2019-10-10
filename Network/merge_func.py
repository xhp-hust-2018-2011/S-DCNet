# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

# merge low2high
def count_merge_low2high_batch(clow,chigh):
    '''
    Inputs must have 4 dim, b*1*h*w
    '''
    IF_gpu = torch.cuda.is_available()
    IF_ret_gpu = (clow.device.type == 'cuda')
    rate = int(chigh.size()[-1]/clow.size()[-1])
    norm = 1/(float(rate)**2)
    cl2h = torch.zeros(chigh.size())
    if IF_gpu:
        clow,chigh,cl2h = clow.cuda(),chigh.cuda(),cl2h.cuda() 

    # b,c,h,w = clow.size()
    for rx in range(rate):
        for ry in range(rate):
            cl2h[:,:,rx::rate,ry::rate] = clow*norm     

    if not IF_ret_gpu: # return as the input device
        cl2h = cl2h.cpu()
    
    return cl2h

if __name__ == '__main__':
    clow = torch.Tensor([4.0])
    chigh = torch.Tensor([[2.0,3.0],[4.0,5.0]])
    clow,chigh = clow.reshape(1,1,1,1),chigh.reshape(1,1,2,2)
    cl2h = count_merge_low2high_batch(clow,chigh)

    print(clow)
    print(chigh)
    print(cl2h)
