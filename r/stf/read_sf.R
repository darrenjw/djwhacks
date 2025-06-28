#!/usr/bin/env Rscript

## read_sf.R
## read a spatio-temporal gpkg file from disk and convert to stars

if (!require("pacman")) install.packages("pacman")
pacman::p_load("sf", "stars")

rural_sf = st_read("air_rural.gpkg")

rural_stars = st_as_stars(rural_sf, dims=c("geom", "time")) # slow!

plot(rural_stars)
image(rural_stars$PM10)


## eof

