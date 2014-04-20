#!/usr/bin/env python

from myMinecraft import *

mc.setBlock(20,20,20,block.COBBLESTONE)
mc.setBlock(minecraft.Vec3(21,21,21),block.DIRT)

line(20,20,20,1,0,0,30,block.COBBLESTONE)
line(25,25,25,0,0,1,5,block.WOOL)

rect(35,35,35,10,5,block.DIRT)

square(40,40,40,10,block.DIRT)

pyramid(90,6,120,10,block.COBBLESTONE)

circle(40,20,40,20,block.DIRT)
circle(40,20,40,10,block.DIRT)
cylinder(40,20,40,5,20,block.COBBLESTONE)


# eof


