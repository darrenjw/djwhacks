#!/usr/bin/env python3
# regress.py
# simple (nonlinear) regression example with pytorch

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# "unknown" 2d functional (to try and learn) - not vectorised
# expects a vector of length 2 - returns a scalar
def truth(x):
    return np.cos(10*np.linalg.norm(x))/(1 + np.linalg.norm(x))

N = 100 # grid resolution for training data

def make_data(N):
    xg = np.arange(-1, 1, 2/N)
    xm, ym = np.meshgrid(xg, xg)
    xv = xm.reshape((N*N, 1))
    yv = ym.reshape((N*N, 1))
    xx = np.concatenate([xv, yv], 1)
    fv = np.apply_along_axis(truth, 1, xx)
    fm = fv.reshape((N, N))
    plt.imshow(fm)
    plt.savefig("truth.pdf")
    return (torch.tensor(xx, dtype=torch.float),
            torch.tensor(fv, dtype=torch.float))

x_all, y_all = make_data(N)
x_all = x_all * x_all # manually engineer the features... CHEATING

# Specify the architecture of the required neural network here
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
            )

    def forward(self, x):
        return self.linear_relu_stack(x)

model = NeuralNetwork()
print(model)

# some utils before starting main loop

def write_frame(file_name, model):
    mat = model(x_all).detach().numpy().reshape((N, N))
    mx = np.max(mat)
    mn = np.min(mat)
    imat = np.uint8((mat-mn)*255//(mx-mn))
    img = Image.fromarray(imat)
    img.save(file_name)

def vec_stats(vec, label):
    vec = vec.detach().numpy()
    length = len(vec)
    mn = np.min(vec)
    mx = np.max(vec)
    mean = np.mean(vec)
    sd = np.std(vec)
    print(f"{label} - len: {length}, mean: {mean:03f}, sd: {sd:03f}, min: {mn:03f}, max: {mx:03f}")

def compare(model, x, y):
    vec_stats(y, "true")
    pred_y = model(x)
    #grads = g_loss(model, x, y)
    #grads = np.asarray(grads.layer2.weight).reshape(-1)
    vec_stats(pred_y, "pred")
    vec_stats(y - pred_y, "error")
    vec_stats((y - pred_y)**2, "squared error")
    #vec_stats(grads, "l2 grads")
    

print("Try to optimise...")
#########################################
learning_rate = 3e-3
batch_size = 128
epochs = 100000
loss_fn = nn.MSELoss()
#########################################
steps = epochs*x_all.shape[0]//batch_size
print(f"{epochs} epochs requires {steps} steps with a bs of {batch_size}")
optim = torch.optim.SGD(model.parameters(),
                        lr=learning_rate, momentum=0.9)

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

# main training loop
iter_data = dataloader(batch_size)
model.train() # not sure what this does
fs = open('loss.csv', 'w')
fs.write("it,loss\n")
for i, (xb, yb) in zip(range(steps), iter_data):
    pred = model(xb)
    loss = loss_fn(pred, yb)
    loss.backward()
    optim.step()
    optim.zero_grad()
    if (i % 10000 == 0):
        print(i, steps-i, loss) # check progress
        compare(model, x_all, y_all)
        fs.write(f"{i},{loss}\n")
        f = i//10000
        write_frame(f"adam{f:05d}.png", model)
fs.close()
    
# how good is the fitted model?
pred = model(x_all).detach().numpy()
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


