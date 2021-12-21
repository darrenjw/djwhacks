#!/usr/bin/env python3

# fBm using (approx) random midpoint displacement algorithm

import numpy as np
import matplotlib.pyplot as plt

N = 10
H = 0.85

n = int(2**N) + 1
D = 2 - H

x = np.zeros(n)
step = n - 1
for L in range(N):
    step = step // 2
    for i in range(int(2**L)):
        idx = (2*i + 1)*step
        mean = 0.5*(x[idx-step] + x[idx+step])
        x[idx] = mean + np.random.normal()*(2**(-H*L))

plt.plot(range(len(x)), x)
plt.title(f'fBM (RMD): H = {H}, D = {D}')
plt.show()

