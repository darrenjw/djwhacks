#!/usr/bin/env python3
# fluid.py
# simple 2d fluid sim

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


m = 200 # number of rows
n = 250 # number of columns
t = 100 # number of time steps (not terminal time)
dt = 0.01 # size of time step

rho = 1 # density of fluid
mu = 0.01 # viscosity coefficient


delta_x = 1.0 / m # true height of rectangular region is 1 (assume dx = dy)
rows, cols = np.mgrid[0:m, 0:n]

# first simulate a random but smooth initial velocity field via a squared-exp GP
# simulate periodic GP via spectral synthesis (with inverse DFT)
print("Simulate random initial velocity field")

def gp():
    mat = np.zeros([m, n], dtype=complex)
    for j in range(m//2):
        for k in range(n//2):
            if ((j==0)&(k==0)):
                sd = 0
            else:
                sd = 100*np.exp(-0.1*(j*j + k*k))
            mat[j, k] = complex(np.random.normal(0., sd), np.random.normal(0., sd))
            if (j == 0)&(k > 0):
                mat[j, n-k] = mat[j, k].conjugate()
            elif (k == 0)&(j > 0):
                mat[m-j, k] = mat[j, k].conjugate()
            elif (j > 0)&(k > 0):
                mat[m-j, n-k] = mat[j, k].conjugate()
    mat[m//2, 0] = 0 # really, just need these to be real...
    mat[0, n//2] = 0
    mat[m//2, n//2] = 0
    for j in range(1, m//2):
        for k in range(1, n//2):
            sd = 100*np.exp(-0.1*(j*j + k*k))
            mat[m-j, k] = complex(np.random.normal(0., sd), np.random.normal(0., sd))
            mat[j, n-k] = mat[m-j, k].conjugate()
    mat = np.fft.ifft2(mat)
    return np.real(mat)

vx = gp()
vy = gp()

plt.imshow(vx)
plt.savefig("vx.pdf")
plt.imshow(vy)
plt.savefig("vy.pdf")

print("Initial velocity field sampled")

# some (periodic) helper functions

# use 2-sided central differences (applied to a scalar field, sf)
def dx(sf):
    return (np.roll(sf, 1, 1) - np.roll(sf, -1, 1)) / (2*delta_x)

def dy(sf):
    return (np.roll(sf, 1, 0) - np.roll(sf, -1, 0)) / (2*delta_x)

# discrete laplacian
def lap(sf):
    return (np.roll(sf, 1, 1) + np.roll(sf, -1, 1) +
            np.roll(sf, 1, 0) + np.roll(sf, -1, 0) - 4*sf) / (delta_x*delta_x)

# solve the poisson equation (in 2d, with periodic boundary conditions)
def solve_poisson(rhs):
    fb = sp.fft.fft2(rhs)
    fp = fb * (delta_x*delta_x) / (2*(np.cos(2*math.pi*rows/m) +
                                      np.cos(2*math.pi*cols/n) - 2))
    fp[0,0] = 0
    p = sp.fft.ifft2(fp)
    return np.real(p)



# now think about time-stepping the Navier-Stokes equations
# start with a simple first-order explicit Euler scheme

for i in range(t):
    print(i)
    plt.imshow(vx)
    plt.savefig(f"vx{i:03d}.png")
    b = -rho*(dx(vx)**2 + 2*dx(vy)*dy(vx) + dy(vy)**2)
    p = solve_poisson(b)
    plt.imshow(p)
    plt.savefig(f"p{i:03d}.png")
    rhs_x = (mu*lap(vx) - dx(p))/rho - dx(vx)*vx - dy(vx)*vy
    rhs_y = (mu*lap(vy) - dy(p))/rho - dx(vy)*vx - dy(vy)*vy
    vx = vx + rhs_x*dt
    vy = vy + rhs_y*dt
    
    



# eof

