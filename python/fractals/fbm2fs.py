#!/usr/bin/env python3

# 2d fBm
# Approximate Fourier synthesis approach

import numpy as np
import matplotlib.pyplot as plt

dt = 0.004
H = 0.85

# Derived quantities
t = np.arange(0., 1., dt)
n = len(t)
D = 3 - H
N = n//2 + 1 # number of Fourier components

# Print some summary info
print(f'N={N}, n={n}, H={H}, D={D}')

# Create an empty matrix to fill with the fractal    
mat = np.zeros([n, n])
x, y = np.meshgrid(t, t)

for j in range(N):
    print(f'j={j}')
    for k in range(N):
        if ((j==0)&(k==0)):
            sd = 0
        else:
            sd = (j*j + k*k)**(-(H+1)/2) # beta = 2H+2 for 2d fBm
        rad = np.random.normal(0., sd)
        phase = np.random.uniform(0., 2*np.pi)
        mat = mat + rad*np.sin(phase)*np.sin(2*np.pi*(j*x + k*y))
        mat = mat + rad*np.cos(phase)*np.cos(2*np.pi*(j*x + k*y))
        # need negative k, too
        rad = np.random.normal(0., sd)
        phase = np.random.uniform(0., 2*np.pi)
        mat = mat + rad*np.sin(phase)*np.sin(2*np.pi*(j*x - k*y))
        mat = mat + rad*np.cos(phase)*np.cos(2*np.pi*(j*x - k*y))

# N.B. This would be MUCH faster with a 2d inverse FFT...
        
print("Matrix filled. Rendering...")

# Just extract centre to hide edge artifacts
mat = mat[(n//4):(3*n//4),(n//4):(3*n//4)]

# Render as a surface
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(range(np.shape(mat)[0]), range(np.shape(mat)[1]))
ax.plot_surface(x, y, mat)
plt.title(f'2d fBm via Fourier synthesis: H={H}, D={D}')
plt.show()

# Render as an image
plt.imshow(mat)
plt.show()

# Convert to a grey image and save as a png
from PIL import Image
mx = np.max(mat)
mn = np.min(mat)
imat = np.uint8((mat-mn)*255//(mx-mn))
img = Image.fromarray(imat)
img.save('fBm2df.png')
img.show()

