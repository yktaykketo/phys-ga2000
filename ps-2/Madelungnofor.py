#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:27:43 2023

@author: wangziyao
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
def t2nofor():
    result=0
    L = 4
    nums = range(-L,L+1)
    x, y, z = np.meshgrid(nums, nums, nums)
    result=np.where((x!=0)|(y!=0)|(z!=0),(-1)**abs(x+y+z)/np.sqrt(x**2+y**2+z**2),0).sum()
    print(result)
    return 0
%timeit t2nofor()