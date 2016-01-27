#!/bin/sh
# make-movie.sh

for name in img????.png
do
  short="${name%.*}"
  echo $short
  #pngtopnm "$name" | pnmscale 20 | pnmtopng > "${short}-s.png"
  convert "$name" -scale 1000x1000 -define png:color-type=2 "${short}-s.png"
done

avconv -r 4 -i img%04d-s.png -r 24 movie.mp4


# eof



