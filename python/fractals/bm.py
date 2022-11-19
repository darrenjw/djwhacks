#!/usr/bin/env python3

# Brownian motion

import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
t = np.arange(0., 10., dt)
n = len(t)

dbt = np.random.normal(0.0, np.sqrt(dt), n)
xt = np.cumsum(dbt)

plt.plot(t, xt)
plt.title("Brownian motion (fBm with H=0.5, D=1.5)")
plt.show()
