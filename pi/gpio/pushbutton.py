#!/usr/bin/python
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)

GPIO.setup(11, GPIO.IN)
GPIO.setup(12, GPIO.OUT)
while True:
	if GPIO.input(11):
		GPIO.output(12, True)
	else:
		GPIO.output(12, False)

