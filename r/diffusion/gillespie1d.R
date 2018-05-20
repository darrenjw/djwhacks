## gillespie1d.R

library(smfsb)

StepGillespie1D <- function(N,d) {
    S = t(N$Post - N$Pre)
    v = ncol(S)
    u = nrow(S)
    return(function(x0, t0, deltat, ...) {
        t = t0
        x = x0
        n = dim(x)[2]
        termt = t + deltat
        repeat {
            hr = apply(x,2,function(x){N$h(x, t, ...)})
            hrs = apply(hr,2,sum)
            hrss = sum(hrs)
            hd = x * (d*2)
            hds = apply(hd,2,sum)
            hdss = sum(hds)
            h0 = hrss + hdss
            if (h0 < 1e-10)
                t = 1e+99
            else if (h0 > 1e+07) {
                t = 1e+99
                warning("Hazard too big - terminating!")
            } else
                t = t + rexp(1, h0)
            if (t > termt) return(x)
            if (runif(1,0,h0) < hdss) {
                ## diffuse
                j = sample(n,1,prob=hds) # pick a box
                i = sample(u,1,prob=hd[,j]) # pick a species
                x[i,j] = x[i,j]-1 # decrement chosen box
                if (runif(1)<0.5) {
                    ## left
                    if (j>1)
                        x[i,j-1] = x[i,j-1]+1
                    else
                        x[i,n] = x[i,n]+1
                } else {
                    ## right
                    if (j<n)
                        x[i,j+1] = x[i,j+1]+1
                    else
                        x[i,1] = x[i,1]+1
                }
            } else {
                ## react
                j = sample(n,1,prob=hrs) # pick a box
                i = sample(v,1,prob=hr[,j]) # pick a reaction
                x[,j] = x[,j] + S[,i]
            }
        }
    })
}

simTs1D <- function (x0, t0 = 0, tt = 100, dt = 0.1, stepFun, verb=FALSE, ...) 
{
    N = (tt - t0)%/%dt + 1
    u = nrow(x0)
    n = ncol(x0)
    arr = array(0,c(u,n,N))
    x = x0
    t = t0
    arr[,,1] = x
    if (verb) cat("Steps",rownames(x),"\n")
    for (i in 2:N) {
        if (verb) cat(N-i,apply(x,1,sum),"\n")
        t = t + dt
        x = stepFun(x, t, dt, ...)
        arr[,,i] = x
    }
    arr
}



## test...

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

