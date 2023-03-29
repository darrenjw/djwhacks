#!/usr/bin/env python3
# ml-ga.py
# Maximum likelihood via gradient ascent (using torch)

import os
import pandas as pd
import numpy as np
import torch

print("Maximum likelihood by gradient ascent using pyTorch")

print("First read and process the data (using regular Python)")
df = pd.read_csv("pima.data", sep=" ", header=None)
print(df)
n, p = df.shape
print(n, p)

y = pd.get_dummies(df[7])["Yes"].to_numpy(dtype='float64')
X = df.drop(columns=7).to_numpy()
X = np.hstack((np.ones((n,1)), X))
print(X)
print(y)

print("Now gradient ascent using Torch")
X = torch.from_numpy(X)
y = torch.from_numpy(y)

print(X)
print(y)

def ll(beta):
    return torch.sum(-torch.log(1 + torch.exp(-(2*y - 1)*(X @ beta))))

init = torch.tensor([-9.8, 0.1, 0, 0, 0, 0, 1.8, 0], dtype=torch.float64)
print("Init:")
print(init)
print(ll(init))

def gll(beta):
    beta.requires_grad = True
    ll0 = ll(beta)
    ll0.backward()
    g = beta.grad
    return g

print(gll(init))

def one_step(b0, learning_rate=1e-6):
    return (b0 + learning_rate*gll(b0)).clone().detach()

print(one_step(init))

def ascend(step, init, max_its=10000, tol=1e-7, verb=True):
    def term(state):
        x1, x0, its = state
        return ((its > 0) & (torch.linalg.vector_norm(x1-x0).item() < tol))
    def step_state(state):
        x1, x0, its = state
        x2 = step(x1)
        return [x2, x1, its - 1]
    state = [init, -init, max_its]
    while (term(state) == False):
        state = step_state(state)
    x1, x0, its = state
    if (verb):
        print(str(its) + " iterations remaining")
    return(x1)

print("Running optimiser now...")
opt = ascend(one_step, init)
print("Optimum:")
print(opt)
print(ll(opt).item())
print("Goodbye.")


# eof

