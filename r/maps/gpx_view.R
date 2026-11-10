#!/usr/bin/env Rscript

## gpx_view.R
## view a GPX track as a line on a Stadia map tile

# library(gpx)
library(sf)
library(ggplot2)
library(ggmap)
library(tmaptools)

# get file name from command line
opts = commandArgs(trailingOnly=TRUE)
if (length(opts) != 1)
    stop("Usage: gpx_view.R <file.gpx>\n")
filename = opts[1]

print(st_layers(filename)) # shows layers in the file
track_sf = st_read(filename, layer="track_points")
lines_sf = st_read(filename, layer="tracks")

## map overlay plot
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

