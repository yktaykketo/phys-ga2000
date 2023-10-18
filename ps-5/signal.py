#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:42:10 2023

@author: wangziyao
"""
import numpy as np
import matplotlib.pyplot as plt

#import data
# source: https://stackoverflow.com/questions/46473270/import-dat-file-as-an-array
def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

data = []
with open('signal.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split('|')
        for i in k:
            if is_float(i):
                data.append(float(i)) 

data = np.array(data, dtype='float')
time = data[::2]
signal = data[1::2]

#1 plot the data
plt.plot(time, signal,'.')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.savefig("problem2a.png",dpi=1000)

plt.cla()

def polyfit(order):
    A = np.zeros((len(time), order+1))
    for i in range(order+1):
        A[:, i] = (time/np.max(time))**i
    (u, w, vt) = np.linalg.svd(A, full_matrices=False)
    cn=np.max(w)/np.min(w)
    print(cn)
    return cn

#2 fit with 3 rd polynomial
order=3
A = np.zeros((len(time), order+1))
A[:, 0] = 1.
A[:, 1] = time/np.max(time)
A[:, 2] = (time/np.max(time))**2
A[:, 3] = (time/np.max(time))**3
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(signal)
ym1 = A.dot(c) 
plt.plot(time, signal, '.', label='data')
plt.plot(time, ym1, '.', label='model')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.savefig("problem2b.png",dpi=1000)


#2 fit the data with third order polynomial
plt.cla()

#3 caculate the residual
flat_signal = signal-ym1
plt.plot(time, flat_signal,'.')
plt.xlabel('Time')
plt.ylabel('Residual Signal')
plt.savefig("problem2c.png",dpi=1000)

plt.cla()

#4 try higher order of polynomial


# polyfit(3)
# polyfit(4)
# polyfit(5)
# polyfit(6)
# polyfit(7)
# polyfit(8)

#5 fit signal with sin and cos
# calculate FFT of oscillations

# from scipy.fft import fft, fftfreq
# # sample spacing
# N=len(time)
# T = (time[N-1]-time[0])/(np.max(time))
# yf = fft(flat_signal)
# xf = fftfreq(N, T)[:N//2]
# omega = 2*np.pi*xf[np.argsort(2.0/N * np.abs(yf[0:N//2]))[::-1]][0]

omega=14.8*np.pi
#omega=2*np.pi/((np.max(time)-np.min(time))/2)


A = np.zeros((len(time), 4))
A[:, 0] = 1.
A[:, 1] = time/np.max(time)
A[:, 2] = np.cos(omega*time/np.max(time))
A[:, 3] = np.sin(omega*time/np.max(time))
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
c = ainv.dot(flat_signal)
ym2 = A.dot(c) 
plt.plot(time, signal, '.', label='data')
plt.plot(time, ym1+ym2, '.', label='model')
#plt.plot(time, signal-(ym1+ym2), '.', label='residual')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.savefig("problem2e.png",dpi=1000)