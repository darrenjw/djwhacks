#!/bin/sh
# make-movie.sh

avconv -r 4 -i out%04d.png -b:v 1000k movie.mp4


# eof



