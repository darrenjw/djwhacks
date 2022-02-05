#!/usr/bin/env python3

# Sierpinski triangle

def sierp(level, x1,y1, x2,y2, x3,y3):
    if (level == 0):
        can.polygon([x1,y1, x2,y2, x3,y3],
                         fill="red", outline="black")
    else:
        x12 = (x1+x2)/2; y12 = (y1+y2)/2
        x13 = (x1+x3)/2; y13 = (y1+y3)/2
        x23 = (x2+x3)/2; y23 = (y2+y3)/2
        sierp(level-1, x1,y1, x12,y12, x13,y13)
        sierp(level-1, x2,y2, x23,y23, x12,y12)
        sierp(level-1, x3,y3, x13,y13, x23,y23)

from PIL import Image, ImageDraw

img = Image.new('RGB', (1000, 1000), color='white')
can = ImageDraw.Draw(img)

sierp(6, 500,100, 100,900, 900,900)

img.show()
img.save('sierp.png')

