#!/usr/bin/env python

import minecraft
import block

mc=minecraft.Minecraft.create()
mc.setBlock(20,20,20,block.COBBLESTONE)
mc.setBlock(minecraft.Vec3(21,21,21),block.DIRT)


# eof


