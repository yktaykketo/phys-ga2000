#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


# In[2]:


def f(x):
    return ((x-0.3)**2)*np.exp(x) 


# In[3]:


def s_quad_interp(a, b, c):
    """
    inverse quadratic interpolation
    """
    epsilon = 1e-7 #for numerical stability
    s0 = a*f(b)*f(c) / (epsilon + (f(a)-f(b))*(f(a)-f(c)))
    s1 = b*f(a)*f(c) / (epsilon + (f(b)-f(a))*(f(b)-f(c)))
    s2 = c*f(a)*f(b) / (epsilon + (f(c)-f(a))*(f(c)-f(b)))
    return s0+s1+s2


# In[4]:


def golden_section_search(f=f, astart=None, bstart=None, cstart=None, tol=1.e-16, maxiter=100):
    gsection = (3. - np.sqrt(5)) / 2
    a = astart
    b = bstart
    c = cstart
    niter = 0
    while((np.abs(c - a) > tol) & (niter < maxiter)):
        # split the larger interval
        if((b - a) > (c - b)):
            x = b
            b = b - gsection * (b - a)
        else:
            x = b + gsection * (c - b)
        fb = f(b)
        fx = f(x)        
        if(fb < fx):
            c = x
        else:
            a = b 
            b = x 
        niter += niter     
    return(b)
golden_section_search(f=f,astart=-1,bstart=-0.5,cstart=1)


# In[9]:


def myoptimize(a,b,c):
    #define interval
    tol = 1e-7
    if abs(f(a)) < abs(f(b)):
        a, b = b, a #swap bounds
    c = a
    flag = True
    err = abs(b-a)
    err_list, b_list = [err], [b]
    while err > tol:
        s = s_quad_interp(a,b,c)
        #print(s)
        if ((s >= b))\
            or ((flag == True) and (abs(s-b) >= abs(b-c)))\
            or ((flag == False) and (abs(s-b) >= abs(c-d))):
            s = golden_section_search(f=f, astart=-1, bstart=-0.2 ,cstart=1)
            flag = True
        else:
            flag = False
        c, d = b, c # d is c from previous step
        #if f(a)*f(s) < 0:
        #    b = s
        #else:
        a = s
        if abs(f(a)) < abs(f(b)):
            a, b = b, a #swap if needed
        err = abs(b-a) #update error to check for convergence
        err_list.append(err)
        b_list.append(b)
    return s
a1 = -1.0
b1 = -0.5
c1 = 1.0
optimizer=myoptimize(a1,b1,c1)


# In[13]:


a = -1.0
b = 0.5
c = 1.0
minimizer = optimize.brent(f, brack=(a, b, c), tol=1.0e-16)
print("Scipy:", minimizer)


# In[14]:


print("Brent's:", optimizer)
print("Difference =", abs(optimizer - minimizer))


# In[ ]:




