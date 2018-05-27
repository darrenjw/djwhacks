## cle2d.R

library(smfsb)
library(abind) # will need to add this to smfsb dependencies...

StepCLE2D <- function(N, d, dt=0.01) {
    S = t(N$Post-N$Pre)
    v = ncol(S)
    u = nrow(S)
    sdt = sqrt(dt)
    left <- function(a) {
        m = dim(a)[2]
        abind(a[,2:m,],a[,1,],along=2)
    }
    right <- function(a) {
        m = dim(a)[2]
        abind(a[,m,],a[,1:(m-1),],along=2)
    }
    down <- function(a) {
        n = dim(a)[3]
        abind(a[,,2:n],a[,,1],along=3)
    }
    up <- function(a) {
        n = dim(a)[3]
        abind(a[,,n],a[,,1:(n-1)],along=3)
    }
    laplacian <- function(a) left(a) + right(a) + up(a) + down(a) - 4*a
    rectify <- function(a) {
        a[a<0] = 0 # absorb at 0
        a
    }
    diffuse <- function(a) {
        m = dim(a)[2]
        n = dim(a)[3]
        dwt = array(rnorm(u*m*n,0,sdt),dim=c(u,m,n))
        dwts = array(rnorm(u*m*n,0,sdt),dim=c(u,m,n))
        a = a + d*laplacian(a)*dt + sqrt(d)*(
            sqrt(a+left(a))*dwt - sqrt(a+right(a))*right(dwt) +
            sqrt(a+up(a))*dwts - sqrt(a+down(a))*down(dwts)
        )
        a = rectify(a)
        a
    }
    return(function(x0, t0, deltat, ...) {
        x = x0
        t = t0
        m = dim(x0)[2]
        n = dim(x0)[3]
        termt = t0 + deltat
        repeat {
            x = diffuse(x)
            hr = apply(x, c(2,3), function(x){N$h(x, t, ...)})
            dwt = array(rnorm(v*m*n,0,sdt),dim=c(v,m,n))
            for (i in 1:m) {
                for (j in 1:n) {
                    x[,i,j] = x[,i,j] + S %*% (hr[,i,j]*dt + sqrt(hr[,i,j])*dwt[,i,j])
                }
            }
            x = rectify(x)
            t = t + dt
            if (t > termt)
                return(x)
        }
    })
}

## Test

LVTest <- function() {
    m=150
    n=100
    T=25
    data(spnModels)
    x0=array(0,c(2,m,n))
    dimnames(x0)[[1]]=c("x1","x2")
    x0[,round(m/2),round(n/2)]=LV$M
    stepLV2D = StepCLE2D(LV,c(0.6,0.6),dt=0.05)
    xx = simTs2D(x0,0,T,0.2,stepLV2D,verb=TRUE)
    N = dim(xx)[4]
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
    m=120
    n=100
    T=3
    data(spnModels)
    x0=array(0,c(3,m,n))
    dimnames(x0)[[1]]=c("S","I","R")
    x0[1,,]=100
    x0[2,round(m/2),round(n/2)]=20
    stepSIR2D = StepCLE2D(SIR,c(1,0.5,0),dt=0.05)
    xx = simTs2D(x0,0,T,0.1,stepSIR2D,verb=TRUE)
    N = dim(xx)[4]
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

