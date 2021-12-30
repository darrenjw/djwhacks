#!/usr/bin/env python3

# Fourier synthesis approach, using iFFT

import numpy as np
import matplotlib.pyplot as plt
from math import pi

dt = 0.0001
t = np.arange(0., 1., dt)
H = 0.85
D = 2 - H
n = len(t)

xt = np.zeros(n, dtype=complex)
for k in range(1, n//2):
    sd = k**(-(H+0.5)) # beta = 2H+1
    xt[k] = complex(np.random.normal(0.0, sd, 1), np.random.normal(0.0, sd, 1))
    xt[n-k] = xt[k].conjugate()

x = np.fft.ifft(xt)
x = np.real(x)

plt.plot(t, x)
plt.title(f'fBM (FS): H = {H}, D = {D}')
plt.show()

