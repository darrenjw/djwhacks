#!/usr/bin/env python

import minecraft
import block

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

line(20,20,20,1,0,0,30,block.COBBLESTONE)
line(25,25,25,0,0,1,5,block.WOOL)

rect(35,35,35,10,5,block.DIRT)

square(40,40,40,10,block.DIRT)

pyramid(40,40,40,50,block.COBBLESTONE)




# eof


