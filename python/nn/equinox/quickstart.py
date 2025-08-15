#!/usr/bin/env python3
# quickstart.py
# example from equinox website


import equinox as eqx
import jax

class NeuralNetwork(eqx.Module):
    layers: list
    extra_bias: jax.Array

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        # These contain trainable parameters.
        self.layers = [eqx.nn.Linear(2, 8, key=key1),
                       eqx.nn.Linear(8, 8, key=key2),
                       eqx.nn.Linear(8, 2, key=key3)]
        # This is also a trainable parameter.
        self.extra_bias = jax.numpy.ones(2)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.extra_bias

@jax.jit  # compile this function to make it run fast.
@jax.grad  # differentiate all floating-point arrays in `model`.
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)  # vectorise the model over a batch of data
    return jax.numpy.mean((y - pred_y) ** 2)  # L2 loss

x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
# Example data
x = jax.random.normal(x_key, (100, 2))
y = jax.random.normal(y_key, (100, 2))
model = NeuralNetwork(model_key)
# Compute gradients
grads = loss(model, x, y)
# Perform gradient descent
learning_rate = 0.1
new_model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)

print("Original")
print(model)

print("New")
print(new_model)


# eof


