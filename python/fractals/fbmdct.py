#!/usr/bin/env python3

# Fourier synthesis approach, using iDCT
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import scipy.fftpack

dt = 0.0001
t = np.arange(0., 1., dt)
H = 0.85
D = 2 - H
n = len(t)

xt = np.zeros(n)
for k in range(1, n):
    sd = k**(-(H+0.5)) # beta = 2H+1
    xt[k] = np.random.normal(0.0, sd, 1)

x = scipy.fftpack.idct(xt)

plt.plot(t, x)
plt.title(f'fBM (DCT): H = {H}, D = {D}')
plt.show()

