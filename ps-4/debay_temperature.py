#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:45:23 2023

@author: wangziyao
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import ones,copy,cos,tan,pi,linspace


V = 1000e-6
rho = 6.022e28
thetad = 428
N = 50
kb = 1.380649e-23

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

def f(x):
    return (x**4)*np.exp(x)/((np.exp(x)-1)**2)

def cvT(t):
    b= thetad/t
    x,w = gaussxw(N)
    a = 0
    xp, wp = gaussxwab(N,a,b)
    s = sum(f(xp)*wp) # add them up!
    coef = 9*V*rho*kb*((t/thetad)**3)
    s=coef*s
    return s

x = np.linspace(5,500,1000)
I = [cvT(xi) for xi in x]
plt.plot(x,I)

plt.title("Heat Capacity of Aluminium")
plt.xlabel("Temperature (K)")
plt.ylabel("$C_V$ ($J.K^{-1}$)")
#plt.show()
plt.savefig("problem1.png",dpi=1000)

plt.clf()

for n in range(10,80,10):
    N=n
    I = [cvT(xi) for xi in x]
    plt.plot(x,I,label='N = %s' %N)
    
plt.legend()
plt.title("Heat Capacity of Aluminium")
plt.xlabel("Temperature (K)")
plt.ylabel("$C_V$ ($J.K^{-1}$)")
#plt.show()
plt.savefig("problem1c.png",dpi=1000)