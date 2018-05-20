## gillespie2d.R

library(smfsb)

StepGillespie2D <- function(N,d) {
    S = t(N$Post - N$Pre)
    v = ncol(S)
    u = nrow(S)
    return(function(x0, t0, deltat, ...) {
        t = t0
        x = x0
        m = dim(x)[2]
        n = dim(x)[3]
        termt = t + deltat
        repeat {
            hr = apply(x,c(2,3),function(x){N$h(x, t, ...)})
            hrs = apply(hr,c(2,3),sum)
            hrss = sum(hrs)
            hd = x * (d*4)
            hds = apply(hd,c(2,3),sum)
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
                r = sample(0:(n*m-1),1,prob=as.vector(hds)) # pick a box
                i = 1 + r %% m
                j = 1 + r %/% m
                k = sample(u, 1, prob=hd[,i,j]) # pick a species
                x[k,i,j] = x[k,i,j]-1 # decrement chosen box
                un = runif(1)
                if (un < 0.25) {
                    ## down
                    if (j>1)
                        x[k,i,j-1] = x[k,i,j-1]+1
                    else
                        x[k,i,n] = x[k,i,n]+1
                } else if (un < 0.5) {
                    ## up
                    if (j<n)
                        x[k,i,j+1] = x[k,i,j+1]+1
                    else
                        x[k,i,1] = x[k,i,1]+1
                } else if (un < 0.75) {
                    ## left
                    if (i>1)
                        x[k,i-1,j] = x[k,i-1,j]+1
                    else
                        x[k,m,j] = x[k,m,j]+1
                } else {
                    ## right
                    if (i<m)
                        x[k,i+1,j] = x[k,i+1,j]+1
                    else
                        x[k,1,j] = x[k,1,j]+1
                }
            } else {
                ## react
                r = sample(0:(n*m-1), 1, prob=as.vector(hrs)) # pick a box
                i = 1 + r %% m
                j = 1 + r %/% m
                k = sample(v, 1, prob=hr[,i,j]) # pick a reaction
                x[,i,j] = x[,i,j] + S[,k]
            }
        }
    })
}

simTs2D <- function (x0, t0 = 0, tt = 100, dt = 0.1, stepFun, verb=FALSE, ...) 
{
    N = (tt - t0)%/%dt + 1
    u = dim(x0)[1]
    m = dim(x0)[2]
    n = dim(x0)[3]
    arr = array(0,c(u,m,n,N))
    x = x0
    t = t0
    arr[,,,1] = x
    if (verb) cat("Steps",dimnames(x)[[1]],"\n")
    for (i in 2:N) {
        if (verb) cat(N-i,apply(x,1,sum),"\n")
        if (verb) image(x[1,,])
        t = t + dt
        x = stepFun(x, t, dt, ...)
        arr[,,,i] = x
    }
    arr
}



## test...

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


LVTest()
##SIRTest()


## eof

