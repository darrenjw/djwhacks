# hourly.sh
# hourly cron

# cron:
# 45 * * * * /home/pi/hourly.sh

cd ~/timelapse

find . -name tl-\*.jpg -type f -mmin +60 -delete

ls *.jpg > stills.txt
rm -f timelapse.avi
mencoder -nosound -ovc lavc -lavcopts vcodec=mpeg4:aspect=16/9:vbitrate=8000000 -vf scale=1920:1080 -o timelapse.avi -mf type=jpeg:fps=12 mf://@stills.txt

avconv -i timelapse.avi -vf scale=640:-1 timelapse.mp4

mv timelapse.mp4 /var/www/


# eof 


