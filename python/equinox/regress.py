#!/usr/bin/env python3
# regress.py
# simple (nonlinear) regression example

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# "unknown" 2d functional (to try and learn) - not vectorised
# expects a vector of length 2 - returns a scalar
def truth(x):
    return jnp.cos(10*jnp.linalg.norm(x))/(1 + jnp.linalg.norm(x))

v_truth = jax.vmap(truth) # vectorised version
# accepts a matrix with 2 columns - returns a vector of results

N = 100
x = jnp.arange(-1, 1, 2/N)
xm, ym = jnp.meshgrid(x, x)
xv = xm.reshape((N*N, 1))
yv = ym.reshape((N*N, 1))
xx = jnp.concatenate([xv, yv], 1)
yv = v_truth(xx)
ym = yv.reshape((N, N))

# So we can use xx and yv as the training data
x = xx
y = yv


plt.imshow(ym)
plt.imsave("truth.pdf", ym)


class NeuralNetwork(eqx.Module):
    layers: list
    extra_bias: jax.Array

    def __init__(self, key):
        keys = jax.random.split(key, 5)
        self.layers = [eqx.nn.Linear(2, 8, key=keys[0]),
                       eqx.nn.Linear(8, 12, key=keys[1]),
                       eqx.nn.Linear(12, 12, key=keys[2]),
                       eqx.nn.Linear(12, 8, key=keys[3]),
                       eqx.nn.Linear(8, 1, key=keys[4])]
        self.extra_bias = jax.numpy.ones(1)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.extra_bias

# @jax.grad  # differentiate all floating-point arrays in `model`.
@jax.jit  # compile this function to make it run fast.
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)  # vectorise the model over a batch of data
    return jax.numpy.mean((y - pred_y) ** 2)  # L2 loss

g_loss = jax.grad(loss)

model_key = jax.random.PRNGKey(42)

model = NeuralNetwork(model_key)

learning_rate = 0.01
# plain old steepest gradient descent
while True:
    print(loss(model, x, y))
    grads = g_loss(model, x, y)
    model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)



# eof


