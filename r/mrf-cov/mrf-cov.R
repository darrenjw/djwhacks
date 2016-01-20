# mrf-cov.R
# Covariance functions for a MRF

xsize=100
ysize=80
pc=0.25 # phase transition for values bigger than 0.25

numpix=xsize*ysize

Q=matrix(0,ncol=numpix,nrow=numpix)

idx=function(x,y) (y-1)*xsize + x

for (x in 1:xsize) {
    for (y in 1:ysize) {
        if (x>1) { Q[idx(x,y),idx(x-1,y)]=-pc
                     Q[idx(x-1,y),idx(x,y)]=-pc }
        if (y>1) { Q[idx(x,y),idx(x,y-1)]=-pc
                     Q[idx(x,y-1),idx(x,y)]=-pc }
        if (x<xsize) { Q[idx(x,y),idx(x+1,y)]=-pc
                         Q[idx(x+1,y),idx(x,y)]=-pc }
        if (y<ysize) { Q[idx(x,y),idx(x,y+1)]=-pc
                         Q[idx(x,y+1),idx(x,y)]=-pc }
     }
}

diag(Q)=1

#op=par(mfrow=c(2,2))
#image(Q)

V=solve(Q)
#image(V)

C=matrix(0,ncol=xsize,nrow=ysize)

xc=round(xsize/2)
yc=round(ysize/2)

for (x in 1:xsize) {
    for (y in 1:ysize) {
        C[y,x]=V[idx(xc,yc),idx(x,y)]
    }
}
image(C)

#par(op)

# eof

