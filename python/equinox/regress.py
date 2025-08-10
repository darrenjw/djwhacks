#!/usr/bin/env python3
# regress.py
# simple (nonlinear) regression example

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
from PIL import Image

# "unknown" 2d functional (to try and learn) - not vectorised
# expects a vector of length 2 - returns a scalar
def truth(x):
    return jnp.cos(10*jnp.linalg.norm(x))/(1 + jnp.linalg.norm(x))

v_truth = jax.vmap(truth) # vectorised version
# accepts a matrix with 2 columns - returns a vector of results

N = 100
xg = jnp.arange(-1, 1, 2/N)
xm, ym = jnp.meshgrid(xg, xg)
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

# Specify the architecture of the required neural network here
class NeuralNetwork(eqx.Module):
    layers: list
    extra_bias: jax.Array

    def __init__(self, key):
        keys = jax.random.split(key, 6)
        self.layers = [eqx.nn.Linear(2, 24, key=keys[0]),
                       eqx.nn.Linear(24, 48, key=keys[1]),
                       eqx.nn.Linear(48, 48, key=keys[2]),
                       eqx.nn.Linear(48, 48, key=keys[3]),
                       eqx.nn.Linear(48, 12, key=keys[4]),
                       eqx.nn.Linear(12, 1, key=keys[5])]
        self.extra_bias = jax.numpy.ones(1)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.extra_bias

@jax.jit
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)  # vectorise the model over a batch of data
    return jax.numpy.mean((y - pred_y) ** 2)  # L2/MSE loss

g_loss = jax.grad(loss)
model_key = jax.random.PRNGKey(2) # master RNG seed
model = NeuralNetwork(model_key)

def write_frame(file_name, model):
    mat = jax.vmap(model)(x).reshape((N, N))
    mx = np.max(mat)
    mn = np.min(mat)
    imat = np.uint8((mat-mn)*255//(mx-mn))
    img = Image.fromarray(imat)
    img.save(file_name)

print("Plain old steepest gradient descent...")
#########################################
learning_rate = 0.1
steps = 2
#########################################
for i in range(steps):
    print(i, loss(model, x, y))
    write_frame(f"gd{i:05d}.png", model)
    grads = g_loss(model, x, y)
    model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)

# how good is the fitted model?
pred = jax.vmap(model)(x)
pred_m = pred.reshape((N,N))
plt.imshow(pred_m)
plt.imsave("pred-gd.pdf", pred_m)
# not very!

print("Try a better optimiser (adam), from optax...")
#########################################
learning_rate = 1e-4
batch_size = 256
epochs = 1000
#########################################
steps = epochs*x.shape[0]//batch_size
print(f"{epochs} epochs requires {steps} steps with a bs of {batch_size}")
optim = optax.adam(learning_rate)
opt_state = optim.init(model)

def dataloader(bs):
    dataset_size = x.shape[0]
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = bs
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield (x[batch_perm], y[batch_perm])
            start = end
            end = start + bs

@jax.jit
def advance(model, x, y, opt_state):
    grads = g_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state

# main training loop
iter_data = dataloader(batch_size)
fs = open('loss.csv', 'w')
fs.write("it,loss\n")
for i, (xb, yb) in zip(range(steps), iter_data):
    if (i % 1000 == 0):
        l = loss(model, x, y)
        print(i, steps-i, l) # check progress
        fs.write(f"{i},{l}\n")
        f = i//1000
        write_frame(f"adam{f:05d}.png", model)
    model, opt_state = advance(model, xb, yb, opt_state)
fs.close()
    
# how good is the fitted model?
pred = jax.vmap(model)(x)
pred_m = pred.reshape((N,N))
plt.imshow(pred_m)
plt.imsave("pred-adam.pdf", pred_m)
# not very!

    
# eof


