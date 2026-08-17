#!/usr/bin/env python3
# lambda.py
# Read and simulate the lambda phage model

# Serves as an illustration of the new "smfsb" python package

import smfsb
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(55)

mod = smfsb.mod_to_spn("lambda.mod")
print(mod)
step = mod.step_gillespie()

# create and plot a single realisation of the process
out = smfsb.sim_time_series(rng, mod.m, 0, 100, 0.1, step)
fig, axis = plt.subplots()
for i in range(len(mod.m)):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(mod.p)
fig.savefig("lambda.pdf")

## Now look at the levels of CII at time 60
print("Sample at time 60. Please wait...")
out = smfsb.sim_sample(rng, 1000, mod.m, 0, 60, step)
cii = out[:,mod.p.index("CII")]
fig, axis = plt.subplots()
axis.hist(cii, 30)
axis.set_title("CII at time 60")
plt.savefig("cii.pdf")

## Now look at the _average_ levels of CII up to time 60
print("Looking now at average up to time 60. Please wait...")
n = 2500
v = np.zeros(n)
for i in range(n):
    out = smfsb.sim_time_series(rng, mod.m, 0, 60, 0.1, step)
    v[i] = np.mean(out[:,mod.p.index("CII")])
fig, axis = plt.subplots()
axis.hist(v, 50)
axis.set_title("CII averaged up to time 60")
plt.savefig("ciiA.pdf")


# eof

