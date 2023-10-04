#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:59:23 2023

@author: wangziyao
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import ones,copy,cos,tan,pi,linspace
import math

#Hermite polynomials
def H(n,x):
    if (n==0):      #H_0(x) = 1
        return 1
    elif (n==1):    #H_1(x) = 2x
        return (2*x)
    
    if(n>1):
        return ((2*x*H(n-1,x)) - (2*(n-1)*H(n-2,x)))

#Wave function
def psi(x,n):
    psi=1/(2**n*math.factorial(n)*(np.pi)**0.5)**0.5*np.exp(-0.5*x**2)*H(n,x)
    return psi

x = np.linspace(-4,4,100)
for n in range(4):
    I = psi(x,n)
    plt.plot(x,I,label='n = %s' %n)

plt.title("Quantum Harmonic Oscillator Wavefunctions")
plt.xlabel('x')
plt.ylabel('$\psi(x)$')
plt.legend()
plt.savefig('problem3a.png',dpi=1000)

#N=30 Wavefunctions
plt.clf()

y = np.linspace(-10,10,1000)
J = psi(y,30)
plt.plot(y,J,label="30")
plt.title("Quantum Harmonic Oscillator Wavefunctions")
plt.xlabel('x')
plt.ylabel('$\psi(x)$')
plt.legend()
plt.savefig('problem3b.png',dpi=1000)



#Quantum uncertainty using gaussian quadrature
N=100
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
    xw,w = gaussxw(N)
    return 0.5*(b-a)*xw+0.5*(b+a),0.5*(b-a)*w
def f(z):
    return tan(z)**2*abs(psi(tan(z),5))**2/(cos(z))**2
b = pi/2
xw,w = gaussxw(N)
a = -pi/2
xp, wp = gaussxwab(N,a,b)
s1 = sum(f(xp)*wp) # add them up!
unc1=s1**0.5
print(unc1)

def f2(z):
    return np.exp(z**2)*z**2*abs(psi(z,5))**2

#Quantum uncertainty using Gauss-Hermite quadrature
from scipy.special import roots_hermite
xw,w = roots_hermite(N) # hermite polynomial roots
#xp, wp = 0.5*(b-a)*xw+0.5*(b+a),0.5*(b-a)*w
s2 = sum(f2(xw)*w) # add them up!
unc2=s2**0.5
print(unc2)
