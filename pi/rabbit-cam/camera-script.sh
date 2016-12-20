#!/bin/bash
# camera-script.sh

while [ "true" != "false" ]
do
  DATE=$(date +"%Y-%m-%d_%H:%M:%S")
  raspistill -o timelapse/tl-$DATE.jpg # time-stamped photo
  cp timelapse/tl-$DATE.jpg /var/www/latest.jpg # keep latest image for serving via web
  # raspistill takes around 8 seconds
  # add any extra delay (in seconds) below:
  sleep 7
done

exit 0

#eof

