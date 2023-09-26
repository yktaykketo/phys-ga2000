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
NBi_213 = 10000 # Number of thallium atoms
NBi_209 = 0
NPb = 0 # Number of lead atoms
NTl=0
total=10000
tau_Tl_Pb = 2.2*60 # Half life of thallium in seconds
tau_Pb_Bi = 3.3*60
tau_Bi_213= 46*60
p_Tl_Pb = 1 - 2**(-h/tau_Tl_Pb) # Probability of decay in one step
p_Pb_Bi = 1 - 2**(-h/tau_Pb_Bi) 
p_Bi = 1 - 2**(-h/tau_Bi_213)
tmax = 20000 # Total time

# Lists of plot points 
tpoints = np.arange(0.0,tmax,h) 
Tlpoints = [] 
Pbpoints = []
Bi_213_points = [] 
Bi_209_points = []
total_amount= []

# Main loop 
for t in tpoints: 
    Tlpoints.append(NTl) 
    Pbpoints.append(NPb)
    Bi_213_points.append(NBi_213)
    Bi_209_points.append(NBi_209)
    total_amount.append(total)
    # Calculate the number of atoms that decay
    #Step a
    decay = 0
    for i in range(NPb): 
        if random.random()<p_Pb_Bi: 
            decay += 1 
    NBi_209 += decay 
    NPb -= decay
       
    #Step b
    decay=0
    for i in range(NTl): 
        if random.random()<p_Tl_Pb: 
            decay += 1 
    NTl -= decay 
    NPb += decay
    
    #Step c
    decay1=0
    decay2=0
    for i in range(NBi_213): 
        if random.random()<p_Bi: 
            if random.random()<0.9791:
                decay1+=1
            else:
                decay2+=1               
    NTl += decay2
    NPb += decay1
    NBi_213-= decay1+decay2
    total=NTl+NPb+NBi_209+NBi_213
# Make the graph 
plt.plot(tpoints,Bi_213_points,label="Bi_213")
plt.plot(tpoints,Bi_209_points,label="Bi_209") 
plt.plot(tpoints,Tlpoints,label="Tl") 
plt.plot(tpoints,Pbpoints,label="Pb") 
plt.plot(tpoints,total_amount,label="Total") 

plt.legend(loc='upper left', bbox_to_anchor=(0.7, 0.8))
plt.xlabel ("Time(s)") 
plt.ylabel("Number of atoms") 
plt.show()