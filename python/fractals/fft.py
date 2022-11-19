#!/usr/bin/env python3

# Figure out how to FFT in Python...

import numpy as np
import matplotlib.pyplot as plt

z = np.random.normal(0.0, 1.0, 1000)

zt = np.fft.ifft(z)

m = np.fromfunction(lambda i, j: (1+i+j)**(-1), (10,10))

mt = np.fft.ifft(m)
print(mt)

# eof
