## gillespie2d.R

library(smfsb)

LVTest <- function() {
    m=40
    n=30
    T=50
    data(spnModels)
    x0=array(0,c(2,m,n))
    dimnames(x0)[[1]]=c("x1","x2")
    x0[,round(m/2),round(n/2)]=LV$M
    stepLV2D = StepGillespie2D(LV,c(0.6,0.6))
    xx = simTs2D(x0,0,T,0.2,stepLV2D,verb=TRUE)
    N = dim(xx)[4]
    cat("max pop (of any species in any voxel) is",max(xx),"\n")
    op=par(mfrow=c(1,2))
    image(xx[1,,,N],main="Prey",xlab="Space",ylab="Time")
    image(xx[2,,,N],main="Predator",xlab="Space",ylab="Time")
    par(op)
    library(grid)
    r = xx[2,,,N]/max(xx[2,,,N])
    g = xx[2,,,N]*0
    b = xx[1,,,N]/max(xx[1,,,N])
    col = rgb(r,g,b)
    dim(col)=dim(r)
    grid.newpage()
    grid.raster(col,interpolate=FALSE)    
}

SIRTest <- function() {
    m=40
    n=30
    T=10
    data(spnModels)
    x0=array(0,c(3,m,n))
    dimnames(x0)[[1]]=c("S","I","R")
    x0[1,,]=100
    x0[2,round(m/2),round(n/2)]=20
    stepSIR2D = StepGillespie2D(SIR,c(1,0.5,0))
    xx = simTs2D(x0,0,T,0.1,stepSIR2D,verb=TRUE)
    N = dim(xx)[4]
    cat("max pop (of any species in any voxel) is",max(xx),"\n")
    op=par(mfrow=c(1,3))
    image(xx[1,,,N],main="S",xlab="Space",ylab="Time")
    image(xx[2,,,N],main="I",xlab="Space",ylab="Time")
    image(xx[3,,,N],main="R",xlab="Space",ylab="Time")
    par(op)
    library(grid)
    r = xx[1,,,N]/max(xx[1,,,N])
    g = xx[2,,,N]/max(xx[2,,,N])
    b = xx[3,,,N]/max(xx[3,,,N])
    col = rgb(r,g,b)
    dim(col)=dim(r)
    grid.newpage()
    grid.raster(col,interpolate=FALSE)    
}


##LVTest()
SIRTest()


## eof

