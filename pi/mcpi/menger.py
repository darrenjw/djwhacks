#!/usr/bin/env python

from myMinecraft import *

def menger(x,y,z,l):
  if (l==0):
    setBlock(x,y,z,block.STONE)
  else:
    s=3**(l-1)
    menger(x+0*s,y+0*s,z+0*s,l-1)
    menger(x+0*s,y+0*s,z+1*s,l-1)
    menger(x+0*s,y+0*s,z+2*s,l-1)
    menger(x+0*s,y+1*s,z+0*s,l-1)
    #menger(x+0*s,y+1*s,z+1*s,l-1)
    menger(x+0*s,y+1*s,z+2*s,l-1)
    menger(x+0*s,y+2*s,z+0*s,l-1)
    menger(x+0*s,y+2*s,z+1*s,l-1)
    menger(x+0*s,y+2*s,z+2*s,l-1)
    menger(x+1*s,y+0*s,z+0*s,l-1)
    #menger(x+1*s,y+0*s,z+1*s,l-1)
    menger(x+1*s,y+0*s,z+2*s,l-1)
    #menger(x+1*s,y+1*s,z+0*s,l-1)
    #menger(x+1*s,y+1*s,z+1*s,l-1)
    menger(x+1*s,y+1*s,z+2*s,l-1)
    #menger(x+1*s,y+2*s,z+0*s,l-1)
    menger(x+1*s,y+2*s,z+1*s,l-1)
    menger(x+1*s,y+2*s,z+2*s,l-1)
    menger(x+2*s,y+0*s,z+0*s,l-1)
    #menger(x+2*s,y+0*s,z+1*s,l-1)
    menger(x+2*s,y+0*s,z+2*s,l-1)
    #menger(x+2*s,y+1*s,z+0*s,l-1)
    menger(x+2*s,y+1*s,z+1*s,l-1)
    menger(x+2*s,y+1*s,z+2*s,l-1)
    menger(x+2*s,y+2*s,z+0*s,l-1)
    menger(x+2*s,y+2*s,z+1*s,l-1)
    menger(x+2*s,y+2*s,z+2*s,l-1)

menger(10,10,10,3)



# eof


