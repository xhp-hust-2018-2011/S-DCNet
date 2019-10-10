# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 20:06:33 2018

@author: xhp
"""
# =============================================================================
# 1. Txt tools
# =============================================================================
import os 
import copy

# Func 1.1: write txt file ( append  or rewrite )
def txt_write(file_name,str,mode='a'):
    if not os.path.exists(file_name):
        mode = 'w'

    with open(file_name,mode) as f:
        f.write(str)


