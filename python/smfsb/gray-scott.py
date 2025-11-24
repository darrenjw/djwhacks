#!/usr/bin/env python3

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

gs_sh = """
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=10 s
 Pop:V=1 s
@parameters
 a=0.014
 b=0.054
@reactions
@r=DegradationU
 U -> 
 (a+b)*U
@r=Production
 -> V
 a
@r=DegredationV
 V ->
 a*V
@r=Reaction
 2U + V -> 3U
 U*U*V
"""

gs = jsmfsb.shorthand_to_spn(gs_sh)

M = 200
N = 250
T = 5
x0 = jnp.zeros((2, M, N)) # init U to 0
x0 = x0.at[1,:,:].set(1) # init V to 1
x0 = x0.at[:, int(M / 2), int(N / 2)].set(gs.m)
step_gs_2d = gs.step_cle_2d(jnp.array([1.0, 2.0]), 0.01)
k0 = jax.random.key(42)
x1 = step_gs_2d(k0, x0, 0, T)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(x1[i, :, :])
    axis.set_title(gs.n[i])
    fig.savefig(f"gs_cle_2df{i}.pdf")


# eof
