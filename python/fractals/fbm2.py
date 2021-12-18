#!/usr/bin/env python3
# fbm2.py
# Diamond square algorithm

import numpy as np
import matplotlib.pyplot as plt

# Choose N and H:
N = 10
H = 0.85


# Derived quantities
n = int(2**N) + 1
D = 3 - H

# Print some summary info
print(f'N={N}, n={n}, H={H}, D={D}')

# Create an empty matrix to fill with the fractal    
mat = np.zeros([n, n])
# Accessor function which wraps around (doing on a torus)
def m(r,c):
    if (r < 0):
        r = r + n
    if (c < 0):
        c = c + n
    if (r >= n):
        r = r - n
    if (c >= n):
        c = c - n
    r = r
    c = c
    return(mat[r,c])

step = n-1
for L in range(N):
    print(f'Level {L}')
    step = step//2
    # diamond step
    for i in range(int(2**L)):
        for j in range(int(2**L)):
            r = (2*i+1)*step
            c = (2*j+1)*step
            mean = 0.25*(m(r-step,c-step)+m(r-step,c+step)+m(r+step,c-step)+m(r+step,c+step))
            mat[r,c] = mean + np.random.normal()*(2**(-H*L))
    # square step
    for i in range(int(2**(L+1))+1):
        for j in range(int(2**(L+1))+1):
            if ((i+j)%2 == 1):
                r = i*step
                c = j*step
                mean = 0.25*(m(r,c-step)+m(r,c+step)+m(r-step,c)+m(r+step,c))
                mat[r,c] = mean + np.random.normal()*(2**(-H*L))

print("Matrix filled. Rendering...")

# Just extract centre to hide edge artifacts
mat = mat[(n//4):(3*n//4),(n//4):(3*n//4)]

# Render
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(range(mat.shape[0]), range(mat.shape[1]))
ax.plot_surface(x, y, mat)
plt.title(f'2d fBm via diamond-square: H={H}, D={D}')
plt.show()
