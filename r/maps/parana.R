## gambia.R

library(sp)
library(sf)
library(ggmap)
library(osmdata)
library(geoR)

parana_df = as.data.frame(parana$coords)
parana_df$rainfall = parana$data
parana_sf = st_as_sf(parana_df, coords=c("east", "north"), crs=3035) # UTM zone 22S
parana_sf2 = st_transform(parana_sf, "EPSG:4326") # WGS

parana_bb = st_bbox(parana_sf2) # sf bb
names(parana_bb) = c("left", "bottom", "right", "top") # ggmap-style bb
print(parana_bb)
parana_map = get_map(parana_bb, maptype="stamen_terrain", source="stadia")

ggmap(parana_map) +
    geom_sf(data = parana_sf2, mapping=aes(colour=rainfall),
            inherit.aes=FALSE) +
    labs(x="Longitude", y="Latitude")


## eof

