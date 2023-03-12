#!/bin/sh
# make-movie.sh

rm -f movie*.mp4

#avconv -r 20 -i siv-%06d-s.png movie.mp4
ffmpeg -f image2 -r 10 -pattern_type glob -i 'mrf-*.png' -s 1920:1080 -vcodec libx264 -crf 10 movie.mp4

# make a version that should play on Android devices...
ffmpeg -i movie.mp4 -codec:v libx264 -profile:v main -preset slow -b:v 400k -maxrate 400k -bufsize 800k -vf scale=-1:480 -threads 0 -codec:a libfdk_aac -b:a 128k -pix_fmt yuv420p movie-a.mp4

# Animated GIF...
# ffmpeg -i movie.mp4 -s 200x100 movie.gif


# eof



