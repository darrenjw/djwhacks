#!/usr/bin/env python3
# poisson.py
# code relating to the solving the 1d poisson equation in various ways

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# use a simple (tractable) RHS
x = np.arange(0, 1, 0.001)
b = np.sin(2*math.pi*x) # easy RHS
fig, axis = plt.subplots(4) # set number of subplots here
axis[0].plot(x, b)
axis[0].set_title("RHS")

# first solve with the FFT
fb = sp.fft.fft(b)
fp = fb * (0.001**2) / (2*(np.cos(2*math.pi*x) - 1))
fp[0] = 0
p = sp.fft.ifft(fp)
p = np.real(p)
axis[1].plot(x, p)
axis[1].set_title("FFT solution")

# now using a sparse matrix (for discrete possion equation)
# sparse matrix for discrete laplacian
lap = sp.sparse.diags_array([np.full((1000), -2),
                           np.full((999), 1), np.full((999), 1)],
                            offsets = [0, 1, -1], format = "csr")
lap = lap / (0.001**2) # scale for grid spacing
p = sp.sparse.linalg.spsolve(lap, b)
axis[2].plot(x, p)
axis[2].set_title("Sparse solution")

# 1d FEM solution...
L = sp.sparse.diags_array([np.full((1000), 0.002/0.001/0.001),
                           np.full((999), -1/0.001),
                           np.full((999), -1/0.001)],
                           offsets = [0, 1, -1], format="csr")
M = sp.sparse.diags_array([np.full((1000), 0.002/3),
                           np.full((999), 0.001/6),
                           np.full((999), 0.001/6)],
                           offsets = [0, 1, -1], format="csr")
p = sp.sparse.linalg.spsolve(L, -M@b)
axis[3].plot(x, p)
axis[3].set_title("FEM solution")


# save figure
fig.savefig("poisson.pdf")



# eof
