#!/usr/bin/env python3

# fBm - exact simulation of the process

import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
H = 0.85

t = np.arange(0., 10., dt)
D = 2 - H
n = len(t)

# Variance matrix
m = np.fromfunction(lambda i, j: 0.5*((i*dt)**(2*H) +
                    (j*dt)**(2*H) -
                    abs((i-j)*dt)**(2*H)),
                    (n, n))
# Drop first row and column to remove time zero,
# otherwise matrix is singular
m = np.delete(m, 0, 0)
m = np.delete(m, 0, 1)
# Drop 0 from vector of times, too
t = np.delete(t, 0)

# Draw from multivariate normal using the Cholesky factor
c = np.linalg.cholesky(m)
z = np.random.normal(0.0, 1.0, n-1)
xt = c.dot(z)

plt.plot(t, xt)
plt.title(f'fBM: H = {H}, D = {D}')
plt.show()

