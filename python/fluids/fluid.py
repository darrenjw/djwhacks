#!/usr/bin/env python3
# fluid.py
# simple 2d fluid sim

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image

m = 480 # number of rows
n = 640 # number of columns
t = 3000 # number of frames (not terminal time)
dt = 0.001 # size of time step
num_steps = 5 # number of time steps per frame (written to disk)

rho = 1 # density of fluid
mu = 0.0005 # viscosity coefficient

diff = 0.0001 # tracer diffusion coefficient

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
                sd = 1000*np.exp(-0.1*(j*j + k*k))
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
            sd = 1000*np.exp(-0.1*(j*j + k*k))
            mat[m-j, k] = complex(np.random.normal(0., sd), np.random.normal(0., sd))
            mat[j, n-k] = mat[m-j, k].conjugate()
    mat = np.fft.ifft2(mat)
    return np.real(mat)

# print some stats
def sf_stats(sf, label="Matrix"):
    mean = np.mean(sf)
    mx = np.max(sf)
    mn = np.min(sf)
    print(f"{label}: mean={mean:03f}, min={mn:03f}, max={mx:03f}")

def sf_to_img(sf):
    mx = np.max(sf)
    mn = np.min(sf)
    imat = np.uint8((sf-mn)*255//(mx-mn))
    return Image.fromarray(imat)

    
vx = gp()
sf_stats(vx, "initial vx")
vy = gp()
sf_stats(vy, "initial vy")

plt.imshow(vx)
plt.savefig("vx.pdf")
plt.imshow(vy)
plt.savefig("vy.pdf")

print("Initial velocity field sampled")

# create a tracer species, s
s = np.zeros((m, n))
s[(m//2):(m//2 + 30), (n//2):(n//2 + 30)] = 1

# some (periodic) helper functions

# use 2-sided central differences (applied to a scalar field, sf)
def dx(sf):
    return (np.roll(sf, -1, 1) - np.roll(sf, 1, 1)) / (2*delta_x)

def dy(sf):
    return (np.roll(sf, -1, 0) - np.roll(sf, 1, 0)) / (2*delta_x)

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

# Navier-Stokes (for incompressible flow)
def advance(vx, vy):
    b = -rho*(dx(vx)**2 + 2*dx(vy)*dy(vx) + dy(vy)**2)
    p = solve_poisson(b)
    rhs_x = (mu*lap(vx) - dx(p))/rho - dx(vx)*vx - dy(vx)*vy
    rhs_y = (mu*lap(vy) - dy(p))/rho - dx(vy)*vx - dy(vy)*vy
    vx = vx + rhs_x*dt
    vy = vy + rhs_y*dt
    return vx, vy

# advection-diffusion for tracer
def advance_tracer(s, vx, vy):
    rhs = diff*lap(s) - vx*dx(s) - vy*dy(s)
    s = s + rhs*dt
    return s


for i in range(t):
    print(i)
    for j in range(num_steps):
        vx, vy = advance(vx, vy)
        s = advance_tracer(s, vx, vy)
    si = sf_to_img(s)
    vxi = sf_to_img(vx)
    vyi = sf_to_img(vy)
    Image.merge("RGB", (si, vxi, vyi)).save(f"mnp{i:05d}.png")
    #print(sf_stats(vx, "vx"))
    #print(sf_stats(vy, "vy"))
    

    
    



# eof

