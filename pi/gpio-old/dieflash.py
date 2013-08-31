#!/usr/bin/python
import RPi.GPIO as GPIO
import time
import random

GPIO.setup (11, GPIO.IN)
GPIO.setup (12, GPIO.OUT)
while True:
	if not GPIO.input(11):
		flash = random.randint(1,6)
		while not GPIO.input(11):
			GPIO.output(12,True)
		while flash > 0:
			GPIO.output(12, False)
			time.sleep(0.5)
			GPIO.output(12, True)
			time.sleep(0.5)
			flash -= 1
	else:
		GPIO.output(12,True)
		time.sleep(0.1)

