#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 22:01:29 2023

@author: wangziyao
"""
from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
def quadratic(a,b,c):
    a=Decimal(a)
    b=Decimal(b)
    c=Decimal(c)
    x_1=(-b+(b**2-4*a*c)**Decimal(0.5))/(2*a)
    x_2=(-b-(b**2-4*a*c)**Decimal(0.5))/(2*a)
    x_3=2*c/(-b+(b**2-4*a*c)**Decimal(0.5))
    x_4=2*c/(-b-(b**2-4*a*c)**Decimal(0.5))
    print(x_1,x_2,x_3,x_4)
    return x_1,x_2
