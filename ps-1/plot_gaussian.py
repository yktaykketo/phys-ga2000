# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:54:35 2023

@author: 86137
"""
import numpy as np
import matplotlib.pyplot as plt
s = np.random.normal(0, 3, 1000000)
#x=np.arange(-10,10,0.02)
plt.hist(s,bins=50, range=(-10,10), density=True,histtype='step')
plt.xlabel('X')
plt.ylabel('Probablity Density')
plt.savefig('gaussian.png')
