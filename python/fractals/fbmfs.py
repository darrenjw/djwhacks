#!/usr/bin/env python3

# Fourier synthesis approach

# Currently WRONG??? - need to check the book...

import numpy as np
import matplotlib.pyplot as plt
from math import pi

dt = 0.0001
t = np.arange(0., 1., dt)
H = 0.85
D = 2 - H
n = len(t)
N = 500 # number of Fourier components

xt = np.zeros(n)
for k in range(1,N):
    sd = k**(-1.5*H) # TODO: check this!!!
    xt = xt + np.random.normal(0.0, sd, 1)*np.sin(2*k*pi*t)
    xt = xt + np.random.normal(0.0, sd, 1)*np.cos(2*k*pi*t)

plt.plot(t, xt)
plt.title(f'fBM: H = {H}, D = {D}')
plt.show()

