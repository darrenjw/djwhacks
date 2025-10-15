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


