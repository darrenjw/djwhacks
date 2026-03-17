#!/usr/bin/env python3
# clyap.py
# solve the continuous lyapunov equation in JAX

# solve AX + XA' = LL'

import jax
import jax.numpy as jnp
import jax.scipy as jsp

# enable 64 bit for testing
jax.config.update("jax_enable_x64", True)

# simulate a test A and Q
n = 10
k0 = jax.random.key(7)
k1, k2 = jax.random.split(k0)
A = jax.random.normal(k1, (n, n))
Q = jax.random.normal(k2, (n, n))  # or could be L


# function to test a solution
def test_clyap(a_mat, l_mat, x_mat, verb=True, tol=1.0e-8):
    z_mat = a_mat @ x_mat + x_mat @ a_mat.T - l_mat @ l_mat.T
    n = jnp.linalg.norm(z_mat)
    if verb:
        print(n)
    return n < tol


# simple kronecker based function
@jax.jit
def clyap(Lambda, Sigma):
    n = Lambda.shape[0]
    kron = jnp.kron(jnp.eye(n), Lambda) + jnp.kron(Lambda, jnp.eye(n))
    VinfV = jnp.linalg.solve(kron, Sigma.reshape(n * n))
    Vinf = VinfV.reshape(n, n)
    return Vinf


Xk = clyap(A, Q @ Q.T)
print(test_clyap(A, Q, Xk))


# *********************************************************
# JAX function to solve the continuous lyapunov equation
# AX + XA' = LL'  for X  (hopefully efficiently)
@jax.jit
def clyap_sqrt(a_mat, l_mat):
    RA, UA = jnp.linalg.eig(a_mat)
    FL = jsp.linalg.solve(UA, l_mat.astype(RA.dtype))
    F = FL @ FL.T
    W = RA[:, None] + RA[None, :]
    Y = F / W
    X = UA @ Y @ UA.T
    return X.real


# *********************************************************

Xe = clyap_sqrt(A, Q)
print(test_clyap(A, Q, Xe))
