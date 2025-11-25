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
 Pop:U=1 s
 Pop:V=1 s
@parameters
 a=0.037
 b=0.06
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

M = 500
N = 600
T = 10000
D = 2
diff_base_rate = 0.1

x0 = jnp.zeros((2, M, N)) # init U to 0
x0 = x0.at[1,:,:].set(1) # init V to 1
x0 = x0.at[:, int(M / 2), int(N / 2)].set(gs.m)
step_gs_2d = gs.step_euler_2d(jnp.array([diff_base_rate, D*diff_base_rate]), 0.05)
k0 = jax.random.key(42)
ts = jsmfsb.sim_time_series_2d(k0, x0, 0, T, 20, step_gs_2d, True)
print(ts.shape)
u_stack = []
v_stack = []
for i in range(ts.shape[3]):
    print(f"Processing frame {i} of {ts.shape[3]}")
    plt.imsave(f"gs-U-{i:05d}.png", ts[0,:,:,i])
    plt.imsave(f"gs-V-{i:05d}.png", ts[1,:,:,i])
    u_stack.append(io.imread(f"gs-U-{i:05d}.png"))
    v_stack.append(io.imread(f"gs-V-{i:05d}.png"))
print("Creating animated gifs")
io.mimsave("gs-U.gif", u_stack)
io.mimsave("gs-V.gif", v_stack)



# eof
