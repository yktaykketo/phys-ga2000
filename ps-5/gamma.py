#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:12:01 2023

@author: wangziyao
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import ones,copy,cos,tan,pi,linspace

def integrand(x,A):
    return x**(A-1)*np.exp(-x)

N=200
x = np.linspace(0,5,100)
I1 = [integrand(xi,2) for xi in x]
I2 = [integrand(xi,3) for xi in x]
I3 = [integrand(xi,4) for xi in x]
plt.plot(x,I1,label="a=2")
plt.plot(x,I2,label="a=3")
plt.plot(x,I3,label="a=4")

plt.legend()
plt.title("Integrand")
plt.xlabel("x value")
plt.ylabel("Integrand value")
plt.show()
plt.savefig("problem1a.png",dpi=1000)

def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    xw = np.cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(xw)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*xw*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-xw*p1)/(1-xw*xw)
        dx = p1/dp
        xw -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-xw*xw)*dp*dp)
    return xw,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def gamma(A):
    c=A-1
    b= 1
    a1 = 0
    x,w = gaussxw(N)
    xp, wp = gaussxwab(N,a1,b)
    
    def f(z):
        return (c/(1-z)**2)*np.exp((A-1)*np.log(z*c/(1-z))-(z*c/(1-z)))
    
    s1 = sum(f(xp)*wp) # add them up!
    print(s1)
    
gamma(1.5)
gamma(3)
gamma(6)
gamma(10)