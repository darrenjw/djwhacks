## waldridge.R

library(gpx)
library(sf)
library(ggplot2)
library(ggmap)
library(tmaptools)

## read GPX data
walk = read_gpx("walk.gpx")
tracks = walk$tracks[[1]]

## start with a very simple plot
plot(tracks$Lon, tracks$Lat, type="l", col=4, lwd=2)
points(tracks$Lon[1], tracks$Lat[1], pch=19, col=3)
points(tracks$Lon[dim(tracks)[1]],
       tracks$Lat[dim(tracks)[1]], pch=19, col=2)

## convert to sf object
track_sf = st_as_sf(tracks, coords=c("Longitude", "Latitude"), crs=4326) # WGS84 for GPS data
lines_sf = st_combine(track_sf) %>% st_cast("LINESTRING") # points to line

## now try a map overlay plot
bbox = st_bbox(track_sf)
bbox = bb(bbox, ext=1.3) # grow bb
names(bbox) = c("left", "bottom", "right", "top") # for ggmap
track_map = get_map(bbox, maptype="stamen_terrain", source="stadia")
ggmap(track_map) +
    geom_sf(data=lines_sf, colour="blue", size=1,
            inherit.aes=FALSE) +
    geom_sf(data=track_sf[1,], colour="green",
            inherit.aes=FALSE) +
    geom_sf(data=track_sf[dim(track_sf)[1],], colour="red",
            inherit.aes=FALSE)


## eof

