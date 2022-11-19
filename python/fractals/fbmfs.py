#!/usr/bin/env python3

# Fourier synthesis approach

import numpy as np
import matplotlib.pyplot as plt
from math import pi

dt = 0.0001
t = np.arange(0., 1., dt)
H = 0.85
D = 2 - H
n = len(t)
N = n//2 # number of Fourier components needed

xt = np.zeros(n)
for k in range(1,N):
    sd = k**(-(H+0.5)) # beta = 2H+1
    xt = xt + np.random.normal(0.0, sd, 1)*np.sin(2*k*pi*t)
    xt = xt + np.random.normal(0.0, sd, 1)*np.cos(2*k*pi*t)

plt.plot(t, xt)
plt.title(f'fBM (FS): H = {H}, D = {D}')
plt.show()

