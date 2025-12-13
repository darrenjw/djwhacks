#!/usr/bin/env python3
# dlyap.py
# solve the discrete lyapunov equation in JAX

# solve AXA' - X + Q = 0  for X

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


# function to test a solution
def test_dlyap(a_mat, q_mat, x_mat, verb=True, tol=1.0e-8):
    z_mat = (a_mat @ x_mat @ a_mat.T) - x_mat + q_mat
    n = jnp.linalg.norm(z_mat)
    if verb:
        print(n)
    return n < tol


print("First try a kronecker solution")

@jax.jit
def dlyap_k(a_mat, q_mat):
    n = A.shape[0]
    kron = jnp.eye(n * n) - jnp.kron(a_mat, a_mat)
    xv = jnp.linalg.solve(kron, q_mat.reshape(n * n))
    return xv.reshape(n, n)


Xk = dlyap_k(A, Q)
print(test_dlyap(A, Q, Xk))

print("Next use the sylvester solver")

@jax.jit
def dlyap_s(a_mat, q_mat):
    n = a_mat.shape[0]
    b_mat = jnp.linalg.solve(a_mat + jnp.eye(n), a_mat - jnp.eye(n))
    r_mat = 0.5 * (jnp.eye(n) - b_mat) @ q_mat @ (jnp.eye(n) - b_mat).T
    return jsp.linalg.solve_sylvester(b_mat, b_mat.T, -r_mat)


Xs = dlyap_s(A, Q)
print(test_dlyap(A, Q, Xs))

print("Next use an eigen-decomposition")

@jax.jit
def dlyap_e(a_mat, q_mat):
    n = a_mat.shape[0]
    e_vals, e_vecs = jnp.linalg.eig(a_mat)
    f_mat = jnp.linalg.solve(e_vecs, (jnp.linalg.solve(e_vecs, q_mat.T)).T)
    w_mat = e_vals[:, None]*e_vals[None, :] - 1
    y_mat = -f_mat / w_mat
    return (e_vecs @ y_mat @ e_vecs.T).real

Xe = dlyap_e(A, Q)
print(test_dlyap(A, Q, Xe))

# eof
