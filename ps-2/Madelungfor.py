#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:55:17 2023

@author: wangziyao
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
def t2for():
    result=0
    # 1st step
    print('[x y z]')
    L = 4   
    for x in range(-L,L+1):
        for y in range(-L,L+1):
            for z in range(-L,L+1):
                point_coordinates = np.array([x, y, z])
                print(point_coordinates)
                signs = (-1.)**np.abs(np.sum(point_coordinates))
                if x==0 and y==0 and z==0:
                    continue
                k=1/(x**2+y**2+z**2)**0.5
                result=signs*k+result
    print(result)
    return 0
%timeit t2for()