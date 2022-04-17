#!/usr/bin/env python3

import jax
from jax import grad, jit
import jax.numpy as jnp

import numpy as np

import math


print("Hello")

@jit
def f(x):
    return ( x[0]*x[1]*jnp.sin(x[2]) + jnp.exp(x[0]*x[1]) )/x[2]

#fj = jit(f)
gf = grad(f)

xx = np.array([1.0, 2.0, math.pi/2])

print( f(xx) )
#print( fj(xx) )

print( gf(xx) )
    
print("Goodbye")

# eof

