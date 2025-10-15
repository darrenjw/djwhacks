# Readme

## Getting started with veros

Full documentation: <https://veros.readthedocs.io/en/latest/introduction/get-started.html> 

First create and activate a new virtual environment for veros in the usual way. Then, in the activated environment, do `uv pip install veros`. Then, from an appropriate directory:
```bash
veros copy-setup acc
cd acc
veros run acc.py
```
This may take a few minutes to run, but will output lots of progress along the way. You can edit the settings in `acc.py` to change aspects of the simulation.

To look at the output, `uv pip install xarray matplotlib` and then paste the following in to a python terminal:
```python
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("acc.snapshot.nc", engine="h5netcdf")

# plot surface velocity at the last time step included in the file
u_surface = ds.u.isel(Time=-1, zt=-1)
u_surface.plot.contourf()
plt.savefig("myplot.pdf")
```

### Other setups

Note that there are other [example setups](https://veros.readthedocs.io/en/latest/reference/setup-gallery.html), in addition to `acc`, including `north_atlantic`.


## Using JAX

To use Veros with JAX, create a different virtual environment, and activate it. Then, in the activated environment, first [install JAX](https://docs.jax.dev/en/latest/installation.html). This is hardware-dependent, but if you don't have a usable GPU (ie. you want to run on CPU) it is likely to be
```bash
uv pip install jax
```
and if you do, it is likely to be
```bash
uv pip install jax[cuda13]
```
Once you have installed JAX, install the JAX version of Veros:
```bash
uv pip install veros[jax]
```
You can then use the JAX-enabled Veros in the same way as the regular numpy version.

