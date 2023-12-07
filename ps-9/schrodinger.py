#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import animation
from IPython.display import HTML
from numpy import copy
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter 


# In[2]:


def create_banded_matrix(diagonals):
    return np.diag(diagonals[0]) + np.diag(diagonals[1], k=1) + np.diag(diagonals[2], k=-1)


# In[3]:


#Constant
N=1000  #spacial grid number
L=1e-8  #box length
a=L/N   #spacial grid size
h=1e-18  #time step size

#Constant in schrodinger function
kai=5e+10
sig=1e-10
h_bar = 1.054571817e-34
m=9.109e-31


a1 = 1+0.5 * 1j * h_bar*h / (m * a**2)
a2 = -0.25 * 1j * h_bar*h/ (m * a**2)
b1 = 1.0 - 0.5*1j * h_bar *h / (m * a**2)
b2 = 0.25*1j * h_bar *h / (m * a**2)

diagonalsA = [
    np.full(N,a1),   # elements of the diagonals
    np.full(N-1,a2),      # elements above the diagonals
    np.full(N-1,a2)     # elements below the diagonals
]

diagonalsB = [
    np.full(N,b1),   
    np.full(N-1,b2),     
    np.full(N-1,b2)       
]

A=create_banded_matrix(diagonalsA)
B=create_banded_matrix(diagonalsB)#(1000, 1000)

# Apply hard boundary conditions
A[0, 0] = 1.0
A[0, 1] = 0.0
A[-1, -1] = 1.0
A[-1, -2] = 0.0

B[0, 0] = 1.0
B[0, 1] = 0.0
B[-1, -1] = 1.0
B[-1, -2] = 0.0


# In[4]:


x = np.linspace(0, L, N) #grid (1000,)
psi = np.exp(-((x - L/2.0)**2) / (2.0*sig**2)) * np.exp(1j * kai * x )  #initial psi
v = np.zeros_like(psi)
#v=np.dot(B,psi)


# In[5]:


nstep = 2000 #time steps
q = np.zeros((nstep, N),dtype = 'complex_')
q[0, :] = psi
for i in np.arange(nstep - 1):
    v=np.dot(B,q[i, :])
    q[i + 1, :] = np.linalg.solve(A, v)


# In[9]:


fig, ax = plt.subplots()
plt.title('Crank-Nicolson Solution for 1D Particle in a box')
plt.xlabel('X (m)')
plt.ylabel('$Re(\Psi(x, t))$')

# Initialize the line
line, = ax.plot([], [], lw=2)

x_min, x_max = 0, 1e-8
ax.set_xlim(x_min, x_max)
y_min, y_max = -1, 1
ax.set_ylim(y_min, y_max)

def init():
    line.set_data([], [])
    return (line,)

def frame(i):
    line.set_data(x, np.real(q[i, :]))

    # Autoscale x-axis in each frame
#     ax.relim()
#     ax.autoscale_view()

    return (line,)

anim = animation.FuncAnimation(fig, frame, init_func=init,
                               frames=nstep, interval=40, blit=True)

# Save animation as a GIF
anim.save('animation.gif', writer='pillow')

# Display animation in Jupyter Notebook
HTML(anim.to_html5_video())


# In[ ]:




