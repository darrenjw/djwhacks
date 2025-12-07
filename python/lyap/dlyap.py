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
Q = jax.random.normal(k2, (n, n))
Q = Q * Q.T  # PSD Q


# function to test a solution
def test_dlyap(a_mat, q_mat, x_mat, verb=True, tol=1.0e-8):
    z_mat = (a_mat @ x_mat @ a_mat.T) - x_mat + q_mat
    n = jnp.linalg.norm(z_mat)
    if verb:
        print(n)
    return n < tol


# simple kronecker based function to start with (scales badly)
@jax.jit
def dlyap_k(a_mat, q_mat):
    n = A.shape[0]
    kron = jnp.eye(n * n) - jnp.kron(a_mat, a_mat)
    xv = jnp.linalg.solve(kron, q_mat.reshape(n * n))
    return xv.reshape(n, n)


Xk = dlyap_k(A, Q)
print(test_dlyap(A, Q, Xk))


# *********************************************************
# JAX function to solve the discrete lyapunov equation
# AXA' - X + Q = 0  for X  (efficiently)
@jax.jit
def dlyap(a_mat, q_mat):
    n = a_mat.shape[0]
    b_mat = jnp.linalg.solve(a_mat + jnp.eye(n), a_mat - jnp.eye(n))
    r_mat = 0.5 * (jnp.eye(n) - b_mat) @ q_mat @ (jnp.eye(n) - b_mat).T
    return jsp.linalg.solve_sylvester(b_mat, b_mat.T, -r_mat)


# *********************************************************

Xs = dlyap(A, Q)
print(test_dlyap(A, Q, Xs))

# eof
