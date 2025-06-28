#!/usr/bin/env Rscript

## st2stars.R
## convert a spacetime STFDF object to stars

if (!require("pacman")) install.packages("pacman")
pacman::p_load("spacetime", "sp", "sf", "stars")

data(air)
rural = STFDF(stations, dates, data.frame(PM10 = as.vector(air)))

## crs is coordinate reference system
## 4326 refers to EPSG:4326 which corresponds to WGS84

stfdf_to_sf <- function(stfdf, crs = 4326) {
  df = as.data.frame(stfdf)
  coords = coordinates(stfdf@sp)[rep(1:nrow(stfdf@sp@coords),
                                      times = length(stfdf@time)), ]
  df$x = coords[, 1]
  df$y = coords[, 2]
  df$time = rep(stfdf@time, each = nrow(stfdf@sp@coords))
  sf::st_as_sf(df, coords = c("x", "y"), crs = crs)
}

rural_sf = stfdf_to_sf(rural)
st_write(rural_sf, "air_rural.gpkg", append=FALSE)

rural_stars = st_as_stars(rural_sf, dims=c("geometry", "time")) # slow!

plot(rural_stars)
image(rural_stars$PM10)






## eof

