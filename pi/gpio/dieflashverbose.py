#!/usr/bin/python
import RPi.GPIO as GPIO
import time
import random

GPIO.setup (11, GPIO.IN)
GPIO.setup (12, GPIO.OUT)
while True:
	if not GPIO.input(11):
		print "press"
		flash = random.randint(1,6)
		while not GPIO.input(11):
			GPIO.output(12,True)
		print "release"
		print "flashing "+str(flash)+" times"
		while flash > 0:
			print("flash")
			GPIO.output(12, False)
			time.sleep(0.5)
			GPIO.output(12, True)
			time.sleep(0.5)
			flash -= 1
		print "done"
	else:
		GPIO.output(12,True)
		time.sleep(0.1)

