#!/usr/bin/env python3
# clyap.py
# solve the continuous lyapunov equation in JAX

# solve AX + XA' = Q

import jax
import jax.numpy as jnp

# enable 64 bit for testing
jax.config.update("jax_enable_x64", True)

# simulate a test A and Q
n = 10
k0 = jax.random.key(7)
k1, k2 = jax.random.split(k0)
A = jax.random.normal(k1, (n, n))
Q = jax.random.normal(k2, (n, n))


# function to test a solution
def test_clyap(a_mat, q_mat, x_mat, verb=True, tol=1.0e-8):
    z_mat = a_mat @ x_mat + x_mat @ a_mat.T - q_mat
    n = jnp.linalg.norm(z_mat)
    if verb:
        print(n)
    return n < tol


print("First a kronecker solution")


@jax.jit
def clyap_k(a_mat, q_mat):
    n = a_mat.shape[0]
    kron = jnp.kron(jnp.eye(n), a_mat) + jnp.kron(a_mat, jnp.eye(n))
    x_vec = jnp.linalg.solve(kron, q_mat.reshape(n * n))
    return x_vec.reshape((n, n))


Xk = clyap_k(A, Q)
print(test_clyap(A, Q, Xk))

print("Next an eigen-decomposition solution")


@jax.jit
def clyap_e(a_mat, q_mat):
    e_vals, e_vecs = jnp.linalg.eig(a_mat)
    f_mat = jnp.linalg.solve(e_vecs, (jnp.linalg.solve(e_vecs, q_mat.T)).T)
    w_mat = e_vals[:, None] + e_vals[None, :]
    y_mat = f_mat / w_mat
    return (e_vecs @ y_mat @ e_vecs.T).real


Xe = clyap_e(A, Q)
print(test_clyap(A, Q, Xe))
