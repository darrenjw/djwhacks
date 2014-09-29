#!/bin/bash
# timelapse.sh

while [ "true" != "false" ]
do
  DATE=$(date +"%Y-%m-%d_%H:%M:%S")
  raspistill -o timelapse/tl-$DATE.jpg
  # raspistill takes around 8 seconds
  # add any extra delay (in seconds) below:
  sleep 7
done

exit 0

#eof

