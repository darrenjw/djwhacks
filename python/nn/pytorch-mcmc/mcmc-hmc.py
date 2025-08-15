#!/usr/bin/env python3
# mcmc-hmc.py
# MCMC with HMC (using torch)

import os
import pandas as pd
import numpy as np
import torch

print("MCMC with HMC using pyTorch")

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

print("Now MCMC using Torch")
X = torch.from_numpy(X)
y = torch.from_numpy(y)

print(X)
print(y)

def ll(beta):
    return torch.sum(-torch.log(1 + torch.exp(-(2*y - 1)*(X @ beta))))

prior_sd = torch.tensor([10,1,1,1,1,1,1,1], dtype=torch.float64)
prior_dist = torch.distributions.normal.Normal(0.0, prior_sd)

def lprior(beta):
    return torch.sum(prior_dist.log_prob(beta))

def lpost(beta):
    return ll(beta) + lprior(beta)

init = torch.tensor([-9.8, 0.1, 0, 0, 0, 0, 1.8, 0], dtype=torch.float64)

print("Init:")
print(init)
print(ll(init))
print(lprior(init))
print(lpost(init))

def glp(beta):
    beta = beta.detach()
    beta.requires_grad = True
    ll0 = lpost(beta)
    ll0.backward()
    g = beta.grad
    return g

print(glp(init))

def mhKernel(lpost, rprop):
    def kernel(x):
        prop = rprop(x)
        a = lpost(prop) - lpost(x)
        if (np.log(np.random.rand()) < a):
            x = prop
        return x
    return kernel
        
def hmcKernel(lpi, glpi, eps = 1e-4, l=10, dmm = 1):
    sdmm = np.sqrt(dmm)
    def leapf(q, p):    
        p = p + 0.5*eps*glpi(q)
        for i in range(l):
            q = q + eps*p/dmm
            if (i < l-1):
                p = p + eps*glpi(q)
            else:
                p = p + 0.5*eps*glpi(q)
        return (q, -p)
    def alpi(x):
        (q, p) = x
        return lpi(q) - 0.5*torch.sum((p**2)/dmm)
    def rprop(x):
        (q, p) = x
        return leapf(q, p)
    mhk = mhKernel(alpi, rprop)
    def kern(q):
        d = len(q)
        p = torch.normal(mean=0, std=sdmm)
        return mhk((q, p))[0]
    return kern

def mcmc(init, kernel, thin = 10, iters = 10000, verb = True):
    p = len(init)
    mat = torch.zeros([iters, p])
    x = init
    if (verb):
        print(str(iters) + " iterations")
    for i in range(iters):
        if (verb):
            print(str(i), end=" ", flush=True)
        for j in range(thin):
            x = kernel(x)
        mat[i] = x.detach()
    if (verb):
        print("\nDone.", flush=True)
    return mat

pre = torch.tensor([100.,1.,1.,1.,1.,1.,25.,1.], dtype=torch.float64)

out = mcmc(init, hmcKernel(lpost, glp, eps=1e-3, l=50, dmm=1/pre), thin=10)
print(out)
np.savetxt("out-hmc.tsv", out.numpy(), delimiter='\t')

print("Goodbye.")


# eof

