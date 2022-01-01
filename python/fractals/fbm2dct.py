#!/usr/bin/env python3

# 2d fBm
# Approximate Fourier synthesis approach, using iDCT

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

dt = 0.001
H = 0.85

# Derived quantities
t = np.arange(0., 1., dt)
n = len(t)
D = 3 - H

# Print some summary info
print(f'n={n}, H={H}, D={D}')

# Create an empty matrix to fill with the DCT coefficients
mat = np.zeros([n, n])

for j in range(n):
    for k in range(n):
        if ((j==0)&(k==0)):
            sd = 0
        else:
            sd = (j*j + k*k)**(-(H+1)/2) # beta = 2H+2 for 2d fBm
        mat[j, k] = np.random.normal(0., sd)
            
mat = scipy.fftpack.idctn(mat)

# Just extract centre to hide edge artifacts
#mat = mat[(n//4):(3*n//4),(n//4):(3*n//4)]

# Render as a surface
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(range(np.shape(mat)[0]), range(np.shape(mat)[1]))
ax.plot_surface(x, y, mat)
plt.title(f'2d fBm via Fourier synthesis (DCT): H={H}, D={D}')
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
img.save('fBm2dct.png')
img.show()

