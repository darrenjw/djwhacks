#!/usr/bin/python
import time
import RPi.GPIO as GPIO

GPIO.setup(12, GPIO.OUT)
GPIO.output(12, False) 
time.sleep(3) 
GPIO.output(12, True) 

