#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,rfft
import pandas as pd
from scipy.signal import find_peaks


# In[2]:


rate=44100


# In[3]:


piano = pd.read_csv('piano.txt', header = None).to_numpy()
plt.plot(np.arange(0, len(piano)), piano)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Piano')
plt.savefig('problem1apiano.png')
plt.show()

trumpet = pd.read_csv('trumpet.txt', header = None).to_numpy()
plt.plot(np.arange(0, len(trumpet)), trumpet)
plt.xlabel('time')
plt.title('Trumpet')
plt.ylabel('amplitude')
plt.savefig('problem1atrumpet.png')


# In[5]:


N=len(piano)
T=1/rate
piano=np.loadtxt("piano.txt")
trumpet=np.loadtxt("trumpet.txt")
xf_piano = fftfreq(N, T)[:N//2]
yf_piano = fft(piano)[:N//2]

xf_trumpet = fftfreq(N, T)[:N//2]
yf_trumpet = fft(trumpet)[:N//2]

# find peak
peaks_piano, _ = find_peaks(np.abs(yf_piano),height=0.2*10**8)
peaks_trumpet, _ = find_peaks(np.abs(yf_trumpet),height=0.2*10**8)

# peak frequencies
peak_frequencies_piano = xf_piano[peaks_piano]
peak_frequencies_trumpet = xf_trumpet[peaks_trumpet]

print("piano frequency:", peak_frequencies_piano)
print("trumpet frequency:", peak_frequencies_trumpet)

plt.plot(xf_piano, np.abs(yf_piano), label='Piano')
plt.xlim(0,5000)
plt.plot(peak_frequencies_piano, np.abs(yf_piano[peaks_piano]), 'ro', label='Piano Peaks') #red at peak
plt.title('Piano')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.savefig('problem1bpiano.png')
plt.show()

plt.plot(xf_trumpet, np.abs(yf_trumpet), label='Trumpet')
plt.xlim(0,5000)
plt.plot(peak_frequencies_trumpet, np.abs(yf_trumpet[peaks_trumpet]), 'bo', label='Trumpet Peaks')  # blue at peak
plt.title('Piano')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.savefig('problem1btrumpet.png')


# In[ ]:





# In[ ]:




