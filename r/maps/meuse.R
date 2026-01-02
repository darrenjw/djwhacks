## meuse.R

library(sp)
library(sf)
library(ggmap)
library(osmdata)

data(meuse)

meuse_sf = st_as_sf(meuse, coords=c("x", "y"), crs=28992)
meuse_sf2 = st_transform(meuse_sf, "EPSG:4326") # WGS
#meuse_sf2 = st_transform(meuse_sf, "EPSG:3857") # Web Mercator

meuse_bb = st_bbox(meuse_sf2) # sf bb
names(meuse_bb) = c("left", "bottom", "right", "top") # ggmap-style bb
meuse_map = get_map(meuse_bb, maptype="stamen_terrain", source="stadia")

ggmap(meuse_map) +
    geom_sf(data = meuse_sf2, mapping=aes(colour=zinc),
            inherit.aes=FALSE) +
    labs(x="Longitude", y="Latitude")


## eof

