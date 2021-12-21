#!/usr/bin/env python3

# 2d fBm
# Approximate Fourier synthesis approach

# Currently INCORRECT!!!


import numpy as np
import matplotlib.pyplot as plt

dt = 0.002
H = 0.85
N = 60 # number of Fourier components

# Derived quantities
t = np.arange(0.,1.,dt)
n = len(t)
D = 3 - H

# Print some summary info
print(f'N={N}, n={n}, H={H}, D={D}')

# Create an empty matrix to fill with the fractal    
mat = np.zeros([n, n])
x, y = np.meshgrid(t, t)

for j in range(1,N):
    print(f'j={j}')
    for k in range(1,N):
        sd = (j**(-1.5*H))*(k**(-1.5*H)) # TODO: check this!!!!
        mat = mat + np.random.normal(0.,sd,1)*np.sin(2*np.pi*(j*x + k*y))
        mat = mat + np.random.normal(0.,sd,1)*np.cos(2*np.pi*(j*x + k*y))

print("Matrix filled. Rendering...")

# Just extract centre to hide edge artifacts
#mat = mat[(n//4):(3*n//4),(n//4):(3*n//4)]

# Render as a surface
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, mat)
plt.title(f'2d fBm via diamond-square: H={H}, D={D}')
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

