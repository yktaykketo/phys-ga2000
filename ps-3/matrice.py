#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:58:45 2023

@author: wangziyao
"""

from numpy import zeros
import numpy as np
import time
import numpy.matlib 
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d

N=100
runtime1=[]
runtime2=[]

x=np.arange(N)

for n in range(N):
    A=np.ones((n,n))
    B=np.ones((n,n))
    C=np.zeros((n,n))
    time1 = time.time()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j]+=A[i,k]*B[k,j]
    time2 = time.time()
    t1=time2-time1
    runtime1.append(t1)
    
    time3 = time.time()
    D=np.dot(A,B)
    time4 = time.time()
    t2=time4-time3
    runtime2.append(t2)
    #print(runtime2)
coeff = polyfit(x, runtime1, 3)
print(coeff)
p = plt.plot(x, coeff[0] * x**3 + coeff[1]*x**2+coeff[2]*x+coeff[3], marker='*',label="Cubic Polynomial Fitting")
plt.plot(x,runtime1,color='r',label="Explicit Function")
plt.plot(x,runtime2,color='b',label="Dot() Method")
plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.8))
plt.xlabel ("Matrix Size N") 
plt.ylabel("Times(s)") 
plt.show()