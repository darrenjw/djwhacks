#!/usr/bin/env python3

# Mandelbrot set

from math import sqrt
from math import log

csize=1000
msize=3.0
z0 = -2.0-1.5j
maxIt = 300

def mandel(c):
    z = 0+0j
    for i in range(maxIt):
        z = z*z + c
        if (abs(z) > 2):
            break
    if (i < maxIt-1):
        return(i)
    else:
        return(-1)

from PIL import Image, ImageDraw
    
img = Image.new('RGB', (1000, 1000), color='white')
can = ImageDraw.Draw(img)
    
print("Mandelbrot set. This may take a few seconds to run...")
    
for x in range(csize):
    for y in range(csize):
        c = z0 + complex(msize*x/csize, msize*y/csize)
        its = mandel(c)
        if (its == -1):
            can.point([x,y], fill="#222222")
        else:
            shade = round(255*log(its+1)/log(maxIt+1))
            col = "#%02x%02x%02x" % (shade,shade,shade)
            can.point([x,y], fill=col)

img.show()
img.save('mandel.png')

