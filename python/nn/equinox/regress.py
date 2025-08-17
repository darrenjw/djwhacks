#!/usr/bin/env python3
# regress.py
# simple (nonlinear) regression example with Equinox

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
from PIL import Image

# try enabling 64 bit, just in case...
# jax.config.update("jax_enable_x64", True)


# "unknown" 2d functional (to try and learn) - not vectorised
# expects a vector of length 2 - returns a scalar
def truth(x):
    return jnp.cos(10*jnp.linalg.norm(x))/(1 + jnp.linalg.norm(x))

v_truth = jax.vmap(truth) # vectorised version
# accepts a matrix with 2 columns - returns a vector of results

N = 100 # grid resolution for training data

def make_data(N):
    xg = jnp.arange(-1, 1, 2/N)
    xm, ym = jnp.meshgrid(xg, xg)
    xv = xm.reshape((N*N, 1))
    yv = ym.reshape((N*N, 1))
    xx = jnp.concatenate([xv, yv], 1)
    yv = v_truth(xx)
    ym = yv.reshape((N, N))
    plt.imshow(ym)
    plt.savefig("truth.pdf")
    return xx, yv

x_all, y_all = make_data(N)
x_all = x_all * x_all # manually engineer the features... CHEATING

# Specify the architecture of the required neural network here
class NeuralNetwork(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear
    layer4: eqx.nn.Linear

    def __init__(self, key):
        keys = jax.random.split(key, 4)
        self.layer1 = eqx.nn.Linear(2, 48, key=keys[0])
        self.layer2 = eqx.nn.Linear(48, 48, key=keys[1])
        self.layer3 = eqx.nn.Linear(48, 24, key=keys[2])
        self.layer4 = eqx.nn.Linear(24, 1, key=keys[3])

    def __call__(self, x):
        x = jax.nn.leaky_relu(self.layer1(x))
        x = jax.nn.leaky_relu(self.layer2(x))
        x = jax.nn.leaky_relu(self.layer3(x))
        return self.layer4(x)

@jax.jit
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)  # vectorise the model over a batch of data
    return jnp.mean((y - pred_y) ** 2)  # L2/MSE loss

g_loss = jax.jit(jax.grad(loss))
model_key = jax.random.PRNGKey(2) # master RNG seed
model = NeuralNetwork(model_key)

# some utils before starting main loop

def write_frame(file_name, model):
    mat = jax.vmap(model)(x_all).reshape((N, N))
    mx = np.max(mat)
    mn = np.min(mat)
    imat = np.uint8((mat-mn)*255//(mx-mn))
    img = Image.fromarray(imat)
    img.save(file_name)

def vec_stats(vec, label):
    length = len(vec)
    mn = np.min(vec)
    mx = np.max(vec)
    mean = np.mean(vec)
    sd = np.std(vec)
    print(f"{label} - len: {length}, mean: {mean:03f}, sd: {sd:03f}, min: {mn:03f}, max: {mx:03f}")

def compare(model, x, y):
    vec_stats(y, "true")
    pred_y = jax.vmap(model)(x)
    grads = g_loss(model, x, y)
    grads = np.asarray(grads.layer2.weight).reshape(-1)
    vec_stats(pred_y, "pred")
    vec_stats(y - pred_y, "error")
    vec_stats((y - pred_y)**2, "squared error")
    vec_stats(grads, "l2 grads")
    
print("Plain old steepest gradient descent...")
#########################################
learning_rate = 0.1
steps = 1
#########################################
for i in range(steps):
    print(i, loss(model, x_all, y_all))
    if (i % 10 == 0):
        compare(model, x_all, y_all)
    write_frame(f"gd{i:05d}.png", model)
    grads = g_loss(model, x_all, y_all)
    model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)

# how good is the fitted model?
pred = jax.vmap(model)(x_all)
pred_m = pred.reshape((N,N))
plt.imshow(pred_m)
plt.savefig("pred-gd.pdf")
# not very!

print("Try a better optimiser, (currently sgd - not adam) from optax...")
#########################################
learning_rate = 3e-3
batch_size = 128
epochs = 100000
#########################################
steps = epochs*x_all.shape[0]//batch_size
print(f"{epochs} epochs requires {steps} steps with a bs of {batch_size}")
optim = optax.adam(learning_rate) # choose sgd or adam here
opt_state = optim.init(model)

def dataloader(bs):
    dataset_size = x_all.shape[0]
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = bs
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield (x_all[batch_perm], y_all[batch_perm])
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
    if (i % 10000 == 0):
        l = loss(model, x_all, y_all)
        print(i, steps-i, l) # check progress
        compare(model, x_all, y_all)
        grads = g_loss(model, xb, yb)
        grads = np.asarray(grads.layer2.weight).reshape(-1)
        vec_stats(grads, "l2 batch")
        fs.write(f"{i},{l}\n")
        f = i//10000
        write_frame(f"adam{f:05d}.png", model)
    model, opt_state = advance(model, xb, yb, opt_state)
fs.close()
    
# how good is the fitted model?
pred = jax.vmap(model)(x_all)
pred_m = pred.reshape((N,N))
plt.imshow(pred_m)
plt.savefig("pred-adam.pdf")
plt.figure()
plt.scatter(y_all, pred)
plt.plot([-1, 1.5], [-1, 1.5], 'r')
plt.title("Predicted against truth")
plt.savefig("pred_truth.pdf")
# not very!

    
# eof


