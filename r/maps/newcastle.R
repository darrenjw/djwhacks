## newcastle.R

library(ggmap)
library(osmdata)

ncl_bb = getbb("Newcastle")
ncl_map = get_map(ncl_bb, maptype="stamen_terrain", source="stadia")

ggmap(ncl_map) +
    labs(x="Longitude", y="Latitude")



## eof

