#!/bin/bash

DATE=$(date +"%Y-%m-%d_%H%M")
raspistill -o timelapse/$DATE.jpg

#eof

