#!/usr/bin/env python3

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio as io

gs_sh = """
@model:3.1.1=GS "Gray-Scott model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=1000 s
 Pop:V=1000 s
@parameters
 N=1000
 a=0.037
 b=0.06
@reactions
@r=DegradationU
 U -> 
 (a+b)*U
@r=Production
 -> V
 a*N
@r=DegredationV
 V ->
 a*V
@r=Reaction
 2U + V -> 3U
 U*U*V/(N*N)
"""

gs = jsmfsb.shorthand_to_spn(gs_sh)

M = 1080 # 1920x1080 is full HD res
N = 1920
T = 5000 # Run for 5k time steps (slow!)
D = 2
diff_base_rate = 0.1

x0 = jnp.zeros((2, M, N)) # init U to 0
x0 = x0.at[1,:,:].set(1000) # init V to 1000
x0 = x0.at[:, int(M / 2), int(N / 2)].set(gs.m)
step_gs_2d = gs.step_cle_2d(jnp.array([diff_base_rate, D*diff_base_rate]), 0.05)
k0 = jax.random.key(42)
ts = jsmfsb.sim_time_series_2d(k0, x0, 0, T, 50, step_gs_2d, True)
print(ts.shape)
u_stack = []
v_stack = []
for i in range(ts.shape[3]):
    print(f"Processing frame {i} of {ts.shape[3]}")
    print(f"U mean: {jnp.mean(ts[0,:,:,i]):0.03f}, V mean: {jnp.mean(ts[1,:,:,i]):0.03f}")
    plt.imsave(f"gssh-U-{i:05d}.png", ts[0,:,:,i])
    plt.imsave(f"gssh-V-{i:05d}.png", ts[1,:,:,i])
    u_stack.append(io.v3.imread(f"gssh-U-{i:05d}.png"))
print("Creating animated gifs")
io.mimsave("gssh-U.gif", u_stack)



# eof
