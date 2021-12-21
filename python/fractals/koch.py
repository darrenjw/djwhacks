#!/usr/bin/env python3

# Koch curve

from tkinter import *
from math import sqrt

root = Tk()
c = Canvas(root, width=1000, height=1000)
c.pack()

def koch(level, x1,y1, x2,y2):
    if (level == 0):
        c.create_line(x1,y1, x2,y2, fill="blue", width=2)
    else:
        vx = (x2-x1)/3; vy = (y2-y1)/3
        xa = x1+vx; ya = y1+vy
        xb = x1+2*vx; yb = y1+2*vy
        xc = xa + vx/2 + vy*sqrt(3)/2
        yc = ya - vx*sqrt(3)/2 + vy/2
        koch(level-1, x1,y1, xa,ya)
        koch(level-1, xa,ya, xc,yc)
        koch(level-1, xc,yc, xb,yb)
        koch(level-1, xb,yb, x2,y2)

for i in range(6):
    koch(i, 200, 180*(0.2+i), 800, 180*(0.2+i))

mainloop()
