#!/bin/sh
# make-movie.sh

for name in img????.png
do
  short="${name%.*}"
  echo $short
  pngtopnm "$name" | pnmscale 20 | pnmtopng > "${short}-s.png"
done

avconv -r 4 -i img%04d-s.png -b:v 1000k movie.mp4


# eof



