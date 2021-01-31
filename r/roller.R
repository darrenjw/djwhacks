## roller.R

t = (0:1000)/1000
w = 10 # angular frequency
## spiral out from origin
x1 = t*cos(2*pi*w*t)
y1 = t*sin(2*pi*w*t)
z1 = t*0
## helix along z axis
x2 = cos(2*pi*w*t)
y2 = sin(2*pi*w*t)
z2 = t
## spriral back in to z axis
x3 = (1-t)*cos(2*pi*w*t)
y3 = (1-t)*sin(2*pi*w*t)
z3 = t*0 + 1
## spiral back to origin inside outer helix
x4 = t*(1-t)*cos(2*pi*w*t)
y4 = t*(1-t)*sin(2*pi*w*t)
z4 = 1-t
## glue peices together
x=c(x1,x2,x3,x4)
y=c(y1,y2,y3,y4)
z=c(z1,z2,z3,z4)
## plot using RGL library
library(rgl)
plot3d(x,y,z,type="l",lwd=2,col=3)

## eof
