# hello.py

from time import sleep
from sense_hat import SenseHat
sense = SenseHat()

sense.set_rotation(0)

sense.show_message("Hello, world!",scroll_speed=0.05, text_colour=[255,255,50], back_colour=[0,0,50])
sleep(2)
sense.clear()

sense.set_pixel(0,0,[255,0,0])
sense.set_pixel(0,1,[0,255,0])
sleep(2)
sense.clear()

t=sense.get_temperature() # celcius
p=sense.get_pressure() # millibars
h=sense.get_humidity() # percent
msg = "Temp: {0}, Press: {1}, Humid: {2}".format(round(t,1),round(p),round(h))
sense.show_message(msg)


# eof


