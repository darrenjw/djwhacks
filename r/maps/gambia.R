## gambia.R

library(sp)
library(sf)
library(ggmap)
library(osmdata)
library(geoR)


gambia_sf = st_as_sf(gambia, coords=c("x", "y"), crs=32628) # UTM zone 28N
gambia_sf2 = st_transform(gambia_sf, "EPSG:4326") # WGS

gambia_bb = st_bbox(gambia_sf2) # sf bb
names(gambia_bb) = c("left", "bottom", "right", "top") # ggmap-style bb
gambia_map = get_map(gambia_bb, maptype="stamen_terrain", source="stadia")

ggmap(gambia_map) +
    geom_sf(data = gambia_sf2, mapping=aes(colour=age),
            inherit.aes=FALSE) +
    labs(x="Longitude", y="Latitude")


## eof

