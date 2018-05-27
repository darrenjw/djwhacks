## gillespie1d.R

library(smfsb)

LVTest <- function() {
    N=40
    T=50
    data(spnModels)
    x0=matrix(0,nrow=2,ncol=N)
    rownames(x0)=c("x1","x2")
    x0[,round(N/2)]=LV$M
    stepLV1D = StepGillespie1D(LV,c(0.6,0.6))
    xx = simTs1D(x0,0,T,0.2,stepLV1D,verb=TRUE)
    cat("max pop (of any species in any voxel) is",max(xx),"\n")
    op=par(mfrow=c(1,2))
    image(xx[1,,],main="Prey",xlab="Space",ylab="Time")
    image(xx[2,,],main="Predator",xlab="Space",ylab="Time")
    par(op)
    library(grid)
    r = xx[2,,]/max(xx[2,,])
    g = xx[2,,]*0
    b = xx[1,,]/max(xx[1,,])
    col = rgb(r,g,b)
    dim(col)=dim(r)
    grid.newpage()
    grid.raster(col,interpolate=FALSE)    
}

SIRTest <- function() {
    N=50
    T=10
    data(spnModels)
    x0=matrix(0,nrow=3,ncol=N)
    rownames(x0)=c("S","I","R")
    x0[1,]=100
    x0[2,round(N/2)]=20
    stepSIR1D = StepGillespie1D(SIR,c(1,0.5,0))
    xx = simTs1D(x0,0,T,0.1,stepSIR1D,verb=TRUE)
    cat("max pop (of any species in any voxel) is",max(xx),"\n")
    op=par(mfrow=c(1,3))
    image(xx[1,,],main="S",xlab="Space",ylab="Time")
    image(xx[2,,],main="I",xlab="Space",ylab="Time")
    image(xx[3,,],main="R",xlab="Space",ylab="Time")
    par(op)
    library(grid)
    r = xx[1,,]/max(xx[1,,])
    g = xx[2,,]/max(xx[2,,])
    b = xx[3,,]/max(xx[3,,])
    col = rgb(r,g,b)
    dim(col)=dim(r)
    grid.newpage()
    grid.raster(col,interpolate=FALSE)    
}


LVTest()
##SIRTest()


## eof

