#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:33:32 2023

@author: wangziyao
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import ones,copy,cos,tan,pi,linspace
m=1
N=20
def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def Period(amp):
    def f(x):
        return 1/(amp**4-x**4)**0.5
    b= amp
    x,w = gaussxw(N)
    a = 0
    xp, wp = gaussxwab(N,a,b)
    s = sum(f(xp)*wp) # add them up!
    coef = (8*m)**0.5
    s=coef*s
    return s

x = np.linspace(0,2,200)
I = [Period(xi) for xi in x]
plt.plot(x,I)

plt.title("Periods of Anharmonic Oscillator")
plt.xlabel("Amplitude (m)")
plt.ylabel("T (s)")
#plt.show()
plt.savefig('problem2.png',dpi=1000)