#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def f(xyz, *, s=10, r=28, b=8/3):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    fx = s*(y - x)
    fy = r*x - y - x*z
    fz = x*y - b*z
    return np.array([fx, fy, fz])


# In[3]:


t0=0
t1=50
h = 0.01
num_steps = 5000
xyzs = (0., 1., 0)  # Set initial values
ts = np.arange(t0, t1+h,h)
rs = np.array([xyzs])


# In[4]:


#RK4
for i in range(num_steps):
    k1 = h*f(xyzs)
    k2 = h*f(xyzs + 0.5*k1)
    k3 = h*f(xyzs + 0.5*k2)
    k4 = h*f(xyzs + k3)
    xyzs += (k1 + 2*k2 + 2*k3 + k4)/6
    rs = np.append(rs, np.array([xyzs]), axis=0)
rs


# In[7]:


# Plot y versus t
plt.plot(ts,rs[:,0])
plt.xlabel("Time(s)")
plt.ylabel("X Axis")
plt.title("Lorenz Equations")
plt.savefig('problem2a.png')
plt.show()


# In[8]:


# Plot z versus x
plt.plot(rs[:,0],rs[:,2])
plt.xlabel("X Axis")
plt.ylabel("Z Axis")
plt.title("Strange Attractor")
plt.savefig('problem2b.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




