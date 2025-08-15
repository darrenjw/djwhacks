#!/usr/bin/env python3
# regress-flax.py
# simple (nonlinear) regression example using FLAX

from flax import nnx
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

# Specify the architecture of the required neural network here
class NeuralNetwork(nnx.Module):

    def __init__(self, din, d2, d3, dout, rngs: nnx.Rngs):
        self.norm1 = nnx.BatchNorm(din, rngs=rngs)
        self.layer1 = nnx.Linear(din, d2, rngs=rngs)
        self.norm2 = nnx.BatchNorm(d2, rngs=rngs)
        self.layer2 = nnx.Linear(d2, d3, rngs=rngs)
        self.norm3 = nnx.BatchNorm(d3, rngs=rngs)
        self.layer3 = nnx.Linear(d3, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.layer1(self.norm1(x)))
        x = nnx.relu(self.layer2(self.norm2(x)))
        return self.layer3(self.norm3(x))

@nnx.jit
def loss(model, x, y):
    pred_y = model(x) 
    return jnp.mean((y - pred_y) ** 2)  # L2/MSE loss

model = NeuralNetwork(2, 12, 24, 1, rngs=nnx.Rngs(0))

# some utils before starting main loop

def write_frame(file_name, model):
    mat = model(x_all).reshape((N, N))
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
    pred_y = model(x)
    grads = nnx.grad(loss)(model, x, y)
    grads = grads.layer2.kernel.reshape(-1)
    vec_stats(pred_y, "pred")
    vec_stats(y - pred_y, "error")
    vec_stats((y - pred_y)**2, "squared error")
    vec_stats(grads, "l2 grads")


print("Try adam from optax...")
#########################################
learning_rate = 1e-3
batch_size = 128
epochs = 10000
#########################################
steps = epochs*x_all.shape[0]//batch_size
print(f"{epochs} epochs requires {steps} steps with a bs of {batch_size}")
optim = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

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

@nnx.jit
def advance(model, optimiser, x, y):
    grads = nnx.grad(loss)(model, x, y)
    optimiser.update(model, grads) # in-place updates?!

print("quick test before main training loop")
print(loss(model, x_all, y_all))
advance(model, optim, x_all, y_all)
print(loss(model, x_all, y_all))
grads = nnx.grad(loss)(model, x_all, y_all)
optim.update(model, grads)
print(loss(model, x_all, y_all))
#print(grads.layer2.kernel.reshape(-1))
print("end of quick test")

# main training loop
iter_data = dataloader(batch_size)
fs = open('loss.csv', 'w')
fs.write("it,loss\n")
for i, (xb, yb) in zip(range(steps), iter_data):
    if (i % 10000 == 0):
        l = loss(model, x_all, y_all)
        print(i, steps-i, l) # check progress
        compare(model, x_all, y_all)
        grads = nnx.grad(loss)(model, xb, yb)
        grads = grads.layer2.kernel.reshape(-1)
        vec_stats(grads, "batch grads")
        fs.write(f"{i},{l}\n")
        f = i//10000
        write_frame(f"adam{f:05d}.png", model)
    advance(model, optim, xb, yb)
fs.close()
    
# how good is the fitted model?
pred = model(x_all)
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


