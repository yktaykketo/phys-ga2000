#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:29:13 2023

@author: wangziyao
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

def f(k):
    value=k**2-k
    return value
delta=0.0000000000000001
x=1
real_derivatives=2*x-1
derivatives=(f(x+delta)-f(x))/delta
difference=derivatives-real_derivatives
print(derivatives)
print(real_derivatives)
print(difference)