#!/usr/bin/env python3
# dlyap.py
# solve the discrete lyapunov equation in JAX

# solve AXA' - X + Q = 0  for X (symmetric Q)

import jax
import jax.numpy as jnp
import jax.scipy as jsp

# enable 64 bit for testing
jax.config.update("jax_enable_x64", True)

# simulate a test A and Q
n = 10
k0 = jax.random.key(24)
k1, k2 = jax.random.split(k0)
A = jax.random.normal(k1, (n, n))
Q = jax.random.normal(k1, (n, n))
Q = Q * Q.T # PSD Q

# function to test a solution
def test_dlyap(A, Q, X, verb=True, tol=1.0e-8):
    Z = (A @ X @ A.T) - X + Q
    n = jnp.linalg.norm(Z)
    if verb:
        print(n)
    return (n < tol)

# simple kronecker based function to start with (scales badly)
@jax.jit
def dlyap_k(A, Q):
    n = A.shape[0]
    kron = jnp.eye(n*n) - jnp.kron(A, A)
    xv = jnp.linalg.solve(kron, Q.reshape(n*n))
    return xv.reshape(n, n)

Xk = dlyap_k(A, Q)
print(test_dlyap(A, Q, Xk))


# *********************************************************
# JAX function to solve the discrete lyapunov equation
# AXA' - X + Q = 0  for X  (efficiently)
@jax.jit
def dlyap(A, Q):
    n = A.shape[0]
    B = jnp.linalg.solve(A + jnp.eye(n), A - jnp.eye(n))
    R = 0.5 * (jnp.eye(n) - B) @ Q @ (jnp.eye(n) - B).T
    return jsp.linalg.solve_sylvester(B, B.T, -R)
# *********************************************************

Xs = dlyap(A, Q)
print(test_dlyap(A, Q, Xs))

# eof

