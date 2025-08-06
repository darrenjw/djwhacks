#!/usr/bin/env python3
# fluid-jax.py
# simple 2d fluid sim, using JAX

import math
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

jax.config.update("jax_enable_x64", True)

m = 720 # number of rows
n = 1280 # number of columns
t = 2500 # number of frames required (not terminal time)
dt = 0.0002 # size of time step
num_steps = 30 # number of time steps per frame (written to disk)

rho = 1 # density of fluid
mu = 0.00002 # viscosity coefficient

diff = 0.0001 # tracer diffusion coefficient

delta_x = 1.0 / m # true height of rectangular region is 1 (assume dx = dy)
rows, cols = jnp.mgrid[0:m, 0:n]

# first simulate a random but smooth initial velocity field via a squared-exp GP
# simulate periodic GP via spectral synthesis (with inverse DFT)
print("Simulate random initial velocity field")

# this function is very slow in JAX - needs re-writing in a more JAX-friendly way
# fortunately it is not part of the hot loop
def gp(key0):
    mat = jnp.zeros([m, n], dtype=complex)
    for j in range(m//2):
        for k in range(n//2):
            if ((j==0)&(k==0)):
                sd = 0
            else:
                sd = 10000*jnp.exp(-0.1*(j*j + k*k))
            key0, key1 = jax.random.split(key0)
            mat=mat.at[j, k].set(complex(jax.random.normal(key0)*sd,
                                         jax.random.normal(key1)*sd))
            if (j == 0)&(k > 0):
                mat=mat.at[j, n-k].set(mat[j, k].conjugate())
            elif (k == 0)&(j > 0):
                mat=mat.at[m-j, k].set(mat[j, k].conjugate())
            elif (j > 0)&(k > 0):
                mat=mat.at[m-j, n-k].set(mat[j, k].conjugate())
    mat=mat.at[m//2, 0].set(0) # really, just need these to be real...
    mat=mat.at[0, n//2].set(0)
    mat=mat.at[m//2, n//2].set(0)
    for j in range(1, m//2):
        for k in range(1, n//2):
            sd = 10000*jnp.exp(-0.1*(j*j + k*k))
            key0, key1 = jax.random.split(key0)
            mat=mat.at[m-j, k].set(complex(jax.random.normal(key0)*sd,
                                           jax.random.normal(key1)*sd))
            mat=mat.at[j, n-k].set(mat[m-j, k].conjugate())
    mat = jnp.fft.ifft2(mat)
    return jnp.real(mat)

# just use numpy version for now...
def gp_np():
    mat = np.zeros([m, n], dtype=complex)
    for j in range(m//2):
        for k in range(n//2):
            if ((j==0)&(k==0)):
                sd = 0
            else:
                sd = 10000*np.exp(-0.1*(j*j + k*k))
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
            sd = 10000*np.exp(-0.1*(j*j + k*k))
            mat[m-j, k] = complex(np.random.normal(0., sd), np.random.normal(0., sd))
            mat[j, n-k] = mat[m-j, k].conjugate()
    mat = np.fft.ifft2(mat)
    return np.real(mat)

# compute some stats (for a scalar field, sf)
def sf_stats(sf, label="Matrix"):
    mean = jnp.mean(sf)
    mx = jnp.max(sf)
    mn = jnp.min(sf)
    return f"{label}: mean={mean:03f}, min={mn:03f}, max={mx:03f}"

def sf_to_img(sf):
    mx = np.max(sf)
    mn = np.min(sf)
    imat = np.uint8((sf-mn)*255//(mx-mn))
    return Image.fromarray(imat)

#k0 = jax.random.key(42)
#k1, k2 = jax.random.split(k0)
#vx = gp(k1)
vx = jnp.array(gp_np()) # use numpy version
print(sf_stats(vx, "initial vx"))
#vy = gp(k2)
vy = jnp.array(gp_np()) # use numpy version
print(sf_stats(vy, "initial vy"))

plt.imshow(vx)
plt.savefig("vx.pdf")
plt.imshow(vy)
plt.savefig("vy.pdf")

print("Initial velocity field sampled")

# create a tracer species, s
s = jnp.zeros((m, n))
s = s.at[(m//2):(m//2 + 80), (n//2):(n//2 + 80)].set(1)

# some (periodic) helper functions

# use 2-sided central differences (applied to a scalar field, sf)
@jax.jit
def dx(sf):
    return (jnp.roll(sf, -1, 1) - jnp.roll(sf, 1, 1)) / (2*delta_x)

@jax.jit
def dy(sf):
    return (jnp.roll(sf, -1, 0) - jnp.roll(sf, 1, 0)) / (2*delta_x)

# discrete laplacian
@jax.jit
def lap(sf):
    return (jnp.roll(sf, 1, 1) + jnp.roll(sf, -1, 1) +
            jnp.roll(sf, 1, 0) + jnp.roll(sf, -1, 0)
            - 4*sf) / (delta_x*delta_x)

# solve the poisson equation (in 2d, with periodic boundary conditions)
@jax.jit
def solve_poisson(rhs):
    fb = jnp.fft.fft2(rhs)
    fp = fb * (delta_x*delta_x) / (2*(jnp.cos(2*math.pi*rows/m) +
                                      jnp.cos(2*math.pi*cols/n) - 2))
    fp=fp.at[0,0].set(0)
    p = jnp.fft.ifft2(fp)
    return jnp.real(p)

# now think about time-stepping the Navier-Stokes equations
# start with a simple first-order explicit Euler scheme

# Navier-Stokes (for incompressible flow)
@jax.jit
def advance(vx, vy):
    b = -rho*(dx(vx)**2 + 2*dx(vy)*dy(vx) + dy(vy)**2)
    p = solve_poisson(b)
    rhs_x = (mu*lap(vx) - dx(p))/rho - dx(vx)*vx - dy(vx)*vy
    rhs_y = (mu*lap(vy) - dy(p))/rho - dx(vy)*vx - dy(vy)*vy
    vx = vx + rhs_x*dt
    vy = vy + rhs_y*dt
    return vx, vy

# advection-diffusion for tracer
@jax.jit
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
    Image.merge("RGB", (si, vxi, vyi)).save(f"m{i:05d}.png")
    #print(sf_stats(vx, "vx"))
    #print(sf_stats(vy, "vy"))
    



# eof

