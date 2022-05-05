#!/usr/bin/env python3

# blackjax example from website
# https://github.com/blackjax-devs/blackjax
# https://blackjax-devs.github.io/blackjax/
# https://blackjax-devs.github.io/blackjax/examples/Introduction.html
# https://blackjax-devs.github.io/blackjax/examples/LogisticRegression.html


import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np

import blackjax

observed = np.random.normal(10, 20, size=1_000)
def logprob_fn(x):
  logpdf = stats.norm.logpdf(observed, x["loc"], x["scale"])
  return jnp.sum(logpdf)

# Build the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.array([1., 1.])
nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)

# Initialize the state
initial_position = {"loc": 1., "scale": 2.}
state = nuts.init(initial_position)

# Main MCMC loop    
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state
    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)
    return states

rng_key = jax.random.PRNGKey(0)
print("Running main MCMC loop...")
states = inference_loop(rng_key, nuts.step, state, 10_000)
print("MCMC loop finished")

loc_samples = states.position["loc"]
scale_samples = states.position["scale"]
    
print(loc_samples)
print(scale_samples)

# Trace plots...
import matplotlib.pyplot as plt
fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
ax.plot(loc_samples)
ax.set_xlabel("Samples")
ax.set_ylabel("loc")

ax1.plot(scale_samples)
ax1.set_xlabel("Samples")
ax1.set_ylabel("scale")

plt.show()


# Multiple chains...

num_chains = 8
initial_positions = {"loc": np.ones(num_chains), "scale": 2.0 * np.ones(num_chains)}
initial_states = jax.vmap(nuts.init, in_axes=(0))(initial_positions)

def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains
):
    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return states, states
    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)
    return states

print("Running multiple chains...")
states = inference_loop_multiple_chains(
    rng_key, nuts.step, initial_states, 2_000, num_chains
)
loc_samples = states.position["loc"]
scale_samples = states.position["scale"]
print("Multiple chains finished.")

fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
ax.plot(loc_samples.T.flatten())
ax.set_xlabel("Samples")
ax.set_ylabel("loc")

ax1.plot(scale_samples.T.flatten())
ax1.set_xlabel("Samples")
ax1.set_ylabel("scale")

plt.show()



# eof
