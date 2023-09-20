# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#matplotlib inline
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

f = np.float32(100.98763)
int32bits = f.view(np.int32)
print('{:032b}'.format(int32bits))
#print(float('01000010110010011111100110101011', 2)) # 2
decimal_float = struct.unpack('f', int32bits)[0]
print(decimal_float)
#values_64 = np.float64(100.98763)
#values_32 = np.float32(100.98763)
#diff = values_64 - values_32
#print(values_32)
#print(diff)

# In IEEE standard, we represent 100.98763 as 1100100.111111001101010101010001110101101000110001101

#0 10000101 10010011111100110101011
#0 10000101 1001001111110011010101010100011101011010001100011010010011

