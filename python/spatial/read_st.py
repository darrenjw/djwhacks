#!/usr/bin/env python3

## read_st.py
## read in a spatio-temporal gpkg file and process appropriately

import pandas as pd
import geopandas as gpd

gdf = gpd.read_file("../../r/stf/air_rural.gpkg")
## read in data in "long" format

print(gdf)
print(gdf.shape)
print(gdf.columns)

## reshape as space-wide
cube = gdf.pivot(index="time", columns="sp.ID", values="PM10")
print(cube)
print(cube.shape)


## eof

