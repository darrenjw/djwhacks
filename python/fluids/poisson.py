# poisson.py
# code relating to the solving the 1d poisson equation in various ways

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

x = np.arange(0, 1, 0.001)
b = np.sin(2*math.pi*x) # easy RHS
fig, axis = plt.subplots(3) # set number of subplots here
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
lap = sp.sparse.diags_array([np.full((1000), -2),
                           np.full((999), 1), np.full((999), 1)],
                            offsets = [0, 1, -1], format = "csr")
lap = lap / (0.001**2)
p = sp.sparse.linalg.spsolve(lap, b)
axis[2].plot(x, p)
axis[2].set_title("Sparse solution")

# TODO: add a 1d FEM solution...


# save figure
fig.savefig("poisson.pdf")



# eof
