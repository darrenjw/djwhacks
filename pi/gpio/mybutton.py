#!/usr/bin/python

import time
import RPi.GPIO as GPIO
GPIO.setup(11,GPIO.IN)

while True:
	mybutton=GPIO.input(11)
	if mybutton==False:
		print "Press"
	time.sleep(0.2)


# eof


