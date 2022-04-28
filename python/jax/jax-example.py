#!/usr/bin/env python3

import jax
from jax import grad, jit
import jax.numpy as jnp

import numpy as np
import scipy as sp
import math


print("Hello")

@jit
def f(x):
    return ( x[0]*x[1]*jnp.sin(x[2]) + jnp.exp(x[0]*x[1]) )/x[2]

gf = grad(f)
xx = np.array([1.0, 2.0, math.pi/2])
print( f(xx) )
print( gf(xx) )


# log-reg example

np.random.seed(10009)
num_features = 10
num_points = 100

true_beta = np.random.randn(num_features).astype(jnp.float32)
all_x = np.random.randn(num_points, num_features).astype(jnp.float32)
y = (np.random.rand(num_points) < sp.special.expit(all_x.dot(true_beta))).astype(jnp.int32)
print(y)




print("Goodbye")




# eof

