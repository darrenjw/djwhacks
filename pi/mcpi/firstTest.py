#!/usr/bin/env python

import minecraft
import block
import math

mc=minecraft.Minecraft.create()
mc.setBlock(20,20,20,block.COBBLESTONE)
mc.setBlock(minecraft.Vec3(21,21,21),block.DIRT)

def line(sx,sy,sz,ix,iy,iz,l,b):
  if l>0:
    mc.setBlock(sx,sy,sz,b)
    line(sx+ix,sy+iy,sz+iz,ix,iy,iz,l-1,b)

def rect(sx,sy,sz,lx,lz,b):
  line(sx,sy,sz,1,0,0,lx,b)
  line(sx,sy,sz,0,0,1,lz,b)
  line(sx,sy,sz+lz-1,1,0,0,lx,b)
  line(sx+lx-1,sy,sz,0,0,1,lz,b)

def square(sx,sy,sz,l,b):
  rect(sx,sy,sz,l,l,b)

def pyramid(sx,sy,sz,l,b):
  if (l>0):
    square(sx,sy,sz,l,b)
    pyramid(sx+1,sy+1,sz+1,l-2,b)

def circle(cx,cy,cz,r,b):
  x=r
  y=0
  while (x>=y):
    mc.setBlock(cx+x,cy,cz+y,b)
    mc.setBlock(cx+x,cy,cz-y,b)
    mc.setBlock(cx-x,cy,cz+y,b)
    mc.setBlock(cx-x,cy,cz-y,b)
    mc.setBlock(cx+y,cy,cz+x,b)
    mc.setBlock(cx+y,cy,cz-x,b)
    mc.setBlock(cx-y,cy,cz+x,b)
    mc.setBlock(cx-y,cy,cz-x,b)
    y=y+1
    x=int(round(math.sqrt(r*r-y*y)))

def cylinder(cx,cy,cz,r,h,b):
  if (h>0):
    circle(cx,cy,cz,r,b)
    cylinder(cx,cy+1,cz,r,h-1,b)

line(20,20,20,1,0,0,30,block.COBBLESTONE)
line(25,25,25,0,0,1,5,block.WOOL)

rect(35,35,35,10,5,block.DIRT)

square(40,40,40,10,block.DIRT)

pyramid(90,6,120,10,block.COBBLESTONE)

circle(40,20,40,20,block.DIRT)
circle(40,20,40,10,block.DIRT)
cylinder(40,20,40,5,20,block.COBBLESTONE)


# eof


