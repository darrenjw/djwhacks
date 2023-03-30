#!/usr/bin/env python3
# mcmc-mh-gpu.py
# MCMC with MH (using torch) - GPU version

import os
import pandas as pd
import numpy as np
import torch

print("MCMC with MH using pyTorch (GPU version)")

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


# TODO: try moving everything to the GPU

print("Now MCMC using Torch")
X = torch.from_numpy(X).to("cuda")
y = torch.from_numpy(y).to("cuda")

print(X)
print(y)

def ll(beta):
    return torch.sum(-torch.log(1 + torch.exp(-(2*y - 1)*(X @ beta))))

prior_sd = torch.tensor([10,1,1,1,1,1,1,1], dtype=torch.float64, device="cuda")
prior_dist = torch.distributions.normal.Normal(0.0, prior_sd)

def lprior(beta):
    return torch.sum(prior_dist.log_prob(beta))

def lpost(beta):
    return ll(beta) + lprior(beta)

init = torch.tensor([-9.8, 0.1, 0, 0, 0, 0, 1.8, 0], dtype=torch.float64,
                    device="cuda")
print("Init:")
print(init)
print(ll(init))
print(lprior(init))
print(lpost(init))

# Don't need this for MH, but leave here anyway
def gll(beta):
    beta.requires_grad = True
    ll0 = ll(beta)
    ll0.backward()
    g = beta.grad
    return g

print(gll(init))

def mhKernel(lpost, rprop):
    def kernel(x, ll):
        prop = rprop(x)
        lp = lpost(prop)
        a = lp - ll
        if (np.log(np.random.rand()) < a.item()):
            x = prop
            ll = lp
        return x, ll
    return kernel
        
def mcmc(init, kernel, thin = 10, iters = 10000, verb = True):
    p = len(init)
    ll = -np.inf
    mat = torch.zeros([iters, p], device="cuda")
    x = init
    if (verb):
        print(str(iters) + " iterations")
    for i in range(iters):
        if (verb):
            print(str(i), end=" ", flush=True)
        for j in range(thin):
            x, ll = kernel(x, ll)
        mat[i] = x.detach()
    if (verb):
        print("\nDone.", flush=True)
    return mat

pre = torch.tensor([10.,1.,1.,1.,1.,1.,5.,1.], dtype=torch.float64, device="cuda")

def rprop(beta):
    return beta + 0.02*torch.normal(mean=0, std=pre)

out = mcmc(init, mhKernel(lpost, rprop), thin=100)
print(out)
np.savetxt("out-mh-gpu.tsv", out.cpu().numpy(), delimiter='\t')

print("Goodbye.")


# eof

