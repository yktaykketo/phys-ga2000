#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:38:52 2023

@author: wangziyao
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# Constants 
h = 1.0 # Size of time-step in seconds
tau = 3.053*60 # Half life of thallium in seconds
tmax = 20000 # Total time

# Lists of plot points 
Tlpoints = [] 

a=np.ones(1000)
b=np.random.random(1000)
x=-(tau/np.log(2))*np.log(a-b)
y=np.sort(x)
tpoints = np.arange(1000,0,-1) 


    
# Make the graph 
plt.plot(y,tpoints,color='r',label="Bi_213")
plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.8))
plt.xlabel ("Time(s)") 
plt.ylabel("Number of atoms") 
plt.show()