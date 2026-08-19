#!/usr/bin/env python
# schnakenberg.py
# A Schnakenberg Turing pattern forming system
# Loosely based on the example from visualpde.com:
# https://visualpde.com/mathematical-biology/schnakenberg

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio as io

sch_sh = """
@model:3.1.1=SCH "Schnakenberg model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=1 s
 Pop:V=1 s
@parameters
 a=0.05
 b=0.5
@reactions
@r=DegradationU
 U -> 
 U
@r=ProductionU
 -> U
 a
@r=ProductionV
 -> V
 b
@r=Reaction
 2U + V -> 3U
 U*U*V
"""

sch = jsmfsb.shorthand_to_spn(sch_sh)

#M = 500; N = 600
M = 100; N = 150
T = 10000; D = 30
a = 0.05; b = 0.5
diff_base_rate = 0.001

x0 = jnp.zeros((2, M, N))
x0 = x0.at[0,:,:].set(a+b) # init U
x0 = x0.at[1,:,:].set(b/(a+b)**2) # init V
#x0 = x0.at[0, int(M / 2):(int(M/2)+5), int(N / 2):(int(N/2)+5)].set(1) # increase U at centre
x0 = x0.at[0, int(M / 2), int(N / 2)].set(1) # increase U at centre
step_sch_2d = sch.step_euler_2d(jnp.array([diff_base_rate, D*diff_base_rate]), 0.05)
k0 = jax.random.key(42)
ts = jsmfsb.sim_time_series_2d(k0, x0, 0, T, 20, step_sch_2d, True)
print(ts.shape)
u_stack = []
v_stack = []
for i in range(ts.shape[3]):
    print(f"Processing frame {i} of {ts.shape[3]}")
    plt.imsave(f"sch-U-{i:05d}.png", ts[0,:,:,i])
    plt.imsave(f"sch-V-{i:05d}.png", ts[1,:,:,i])
    u_stack.append(io.imread(f"sch-U-{i:05d}.png"))
    v_stack.append(io.imread(f"sch-V-{i:05d}.png"))
print("Creating animated gifs")
io.mimsave("sch-U.gif", u_stack)
io.mimsave("sch-V.gif", v_stack)






# eof


