#!/usr/bin/env python
# schnaken-stoch.py
# A Schnakenberg Turing pattern forming system - stochastic version

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio as io

sch_sh = """
@model:3.1.1=SCHStoch "Schnakenberg model - stochastic version"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:U=1000 s
 Pop:V=1000 s
@parameters
 Size=1000
 a=0.05
 b=0.5
@reactions
@r=DegradationU
 U -> 
 U
@r=ProductionU
 -> U
 a*Size
@r=ProductionV
 -> V
 b*Size
@r=Reaction
 2U + V -> 3U
 U*U*V/(Size*Size)
"""

sch = jsmfsb.shorthand_to_spn(sch_sh)

M = 400; N = 500
T = 500; D = 100
Size = 1000
a = 0.05; b = 0.5
diff_base_rate = 10.0
dt = 0.0002

x0 = jnp.zeros((2, M, N))
x0 = x0.at[0,:,:].set(Size*(a+b)) # init U
x0 = x0.at[1,:,:].set(Size*b/(a+b)**2) # init V
#x0 = x0.at[0, int(M / 2), int(N / 2)].set(Size) # increase U at centre
step_sch_2d = sch.step_cle_2d(jnp.array([diff_base_rate, D*diff_base_rate]), dt)
k0 = jax.random.key(24)
ts = jsmfsb.sim_time_series_2d(k0, x0, 0, T, 1, step_sch_2d, True)
print(ts.shape)
print("Save final frame")
for i in range(2):
    plt.matshow(ts[i,:,:,-1], cmap='gray_r')
    plt.title(f"Schnakenberg: {sch.p[i]}")
    plt.savefig(f"ch09-sch-stoch-{i}.ps", dpi=600, bbox_inches='tight')
print("Save individual frames...")
u_stack = []
v_stack = []
for i in range(ts.shape[3]):
    print(f"Processing frame {i} of {ts.shape[3]}")
    plt.imsave(f"sch-stoch-U-{i:05d}.png", ts[0,:,:,i])
    plt.imsave(f"sch-stoch-V-{i:05d}.png", ts[1,:,:,i])
    u_stack.append(io.imread(f"sch-stoch-U-{i:05d}.png"))
    v_stack.append(io.imread(f"sch-stoch-V-{i:05d}.png"))
print("Creating animated gifs...")
io.mimsave("sch-stoch-U.gif", u_stack)
io.mimsave("sch-stoch-V.gif", v_stack)






# eof


