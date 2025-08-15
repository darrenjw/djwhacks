#!/usr/bin/env python3

import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as np
import scipy as sp
import math


print("Hello")

print("auto-diff example")

@jit
def f(x):
    return ( x[0]*x[1]*jnp.sin(x[2]) + jnp.exp(x[0]*x[1]) )/x[2]

gf = grad(f)
xx = np.array([1.0, 2.0, math.pi/2])
print( f(xx) )
print( gf(xx) )


print("log-reg example")

np.random.seed(10009)
num_features = 10
num_points = 100

true_beta = np.random.randn(num_features).astype(jnp.float32)
all_x = np.random.randn(num_points, num_features).astype(jnp.float32)
y = (np.random.rand(num_points) < sp.special.expit(all_x.dot(true_beta))).astype(jnp.int32)

print(true_beta)
print(y)

# non-batched
@jit
def log_joint(beta):
    result = 0.
    # Note that no `axis` parameter is provided to `jnp.sum`.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=1.))
    result = result + jnp.sum(-jnp.log(1 + jnp.exp(-(2*y-1) * jnp.dot(all_x, beta))))
    return result

test_beta = np.random.randn(num_features)
print(log_joint(test_beta))
print(log_joint(true_beta))

g1 = grad(log_joint)
print(g1(test_beta))

# pull out linear predictor
@jit
def log_joint2(beta):
    result = 0.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=1.))
    lp = jnp.dot(all_x, beta)
    result = result + jnp.sum(-jnp.log(1 + jnp.exp(-(2*y-1) * lp)))
    return result

print(log_joint2(test_beta))

g2 = grad(log_joint2)
print(g2(test_beta))

# likelihood in more "usual" form
@jit
def log_joint3(beta):
    result = 0.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=1.))
    lp = jnp.dot(all_x, beta)
    result = result + jnp.sum(y*lp - jnp.log(1 + jnp.exp(lp)))
    return result

print(log_joint3(test_beta))

g3 = grad(log_joint3)
print(g3(test_beta))

@jit
def log_lik(beta):
    result = 0.
    lp = jnp.dot(all_x, beta)
    result = result + jnp.sum(y*lp - jnp.log(1 + jnp.exp(lp)))
    return result

print(log_lik(test_beta))

g4 = grad(log_lik)
print(g4(test_beta))

# Compare with hard-coded gradients and hessian...

@jit
def myGrad(beta):
    lp = jnp.dot(all_x, beta)
    p = 1/(1 + jnp.exp(-lp))
    return(jnp.dot(all_x.T, y - p))

print(myGrad(test_beta))

# Manually batched

def batched_log_joint(beta):
    result = 0.
    # Here (and below) `sum` needs an `axis` parameter. At best, forgetting to set axis
    # or setting it incorrectly yields an error; at worst, it silently changes the
    # semantics of the model.
    result = result + jnp.sum(jsp.stats.norm.logpdf(beta, loc=0., scale=1.),
                           axis=-1)
    # Note the multiple transposes. Getting this right is not rocket science,
    # but it's also not totally mindless. (I didn't get it right on the first
    # try.)
    result = result + jnp.sum(-jnp.log(1 + jnp.exp(-(2*y-1) * jnp.dot(all_x, beta.T).T)),
                           axis=-1)
    return result

batch_size = 10
batched_test_beta = np.random.randn(batch_size, num_features)
print(batched_log_joint(batched_test_beta))

# Autobatched with vmap
vmap_batched_log_joint = jax.vmap(log_joint)
print(vmap_batched_log_joint(batched_test_beta))



# Optimise...

from jax import jacfwd, jacrev

def hessian(f):
    return jacfwd(jacrev(f))

hess = hessian(log_joint)
# print(hess(true_beta))
# print(hess(test_beta))


print("Test then true")
print(test_beta)
print(true_beta)

print("Newton method")

current_beta = test_beta
for i in range(50):
    #print(i)
    #print(current_beta)
    step = jsp.linalg.solve(hess(current_beta), g1(current_beta))
    #print(step)
    current_beta = current_beta - 0.1*step

print("Test then estimated then true")
print(test_beta)
print(current_beta)
print(true_beta)

print("Gradient ascent")

current_beta = test_beta
for i in range(100):
    #print(i)
    #print(current_beta)
    step = g1(current_beta)
    #print(step)
    current_beta = current_beta + 0.01*step

print("Test then estimated then true")
print(test_beta)
print(current_beta)
print(true_beta)

print("Method for gradient ascent")

def ga(init_beta):
    current_beta = init_beta
    for i in range(100):
        #print(i)
        #print(current_beta)
        step = g1(current_beta)
        #print(step)
        current_beta = current_beta + 0.01*step
    return(current_beta)

print(ga(test_beta))







print("Goodbye")

# eof

