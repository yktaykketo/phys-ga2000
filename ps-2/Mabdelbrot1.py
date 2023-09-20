#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:54:39 2023

@author: wangziyao
"""

import numpy as np
import matplotlib.pyplot as plt

xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
width, height = 800, 800
img = np.zeros((height, width))
max_iter = 256
escape_radius = 2.0
x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y
mandelbrot = np.zeros((height, width), dtype=int)
Z = np.zeros_like(C, dtype=complex)
for i in range(max_iter):
    mask = np.abs(Z) < escape_radius
    Z[mask] = Z[mask] ** 2 + C[mask]
    mandelbrot += mask
plt.imshow(mandelbrot, extent=(xmin, xmax, ymin, ymax))
plt.colorbar(label='Iterations')
plt.title('Mandelbrot')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
