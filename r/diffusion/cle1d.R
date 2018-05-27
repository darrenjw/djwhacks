## cle1d.R

library(smfsb)

StepCLE1D <- function(N, d, dt=0.01) {
    S = t(N$Post-N$Pre)
    v = ncol(S)
    u = nrow(S)
    sdt = sqrt(dt)
    forward <- function(m) cbind(m[,2:ncol(m)],m[,1])
    back <- function(m) {
        n = ncol(m)
        cbind(m[,n],m[,1:(n-1)])
    }
    laplacian <- function(m) forward(m) + back(m) - 2*m
    rectify <- function(m) {
        m[m<0] = 0 # absorb at 0
        m
    }
    diffuse <- function(m) {
        n = ncol(m)
        noise = matrix(rnorm(n*u,0,sdt),nrow=u)
        m = m + d*laplacian(m)*dt + sqrt(d)*(
            sqrt(m+forward(m))*noise -
            sqrt(m+back(m))*back(noise))
        m = rectify(m)
        m
    }
    return(function(x0, t0, deltat, ...) {
        x = x0
        t = t0
        n = ncol(x0)
        termt = t0 + deltat
        repeat {
            x = diffuse(x)
            hr = apply(x,2,function(x){N$h(x, t, ...)})
            dwt = matrix(rnorm(n*v,0,sdt),nrow=v)
            x = x + S %*% (hr*dt + sqrt(hr)*dwt)
            x = rectify(x)
            t = t + dt
            if (t > termt)
                return(x)
        }
    })
}



## Test

LVTest <- function() {
    N=200
    T=50
    data(spnModels)
    x0=matrix(0,nrow=2,ncol=N)
    rownames(x0)=c("x1","x2")
    x0[,round(N/2)]=LV$M
    stepLV1D = StepCLE1D(LV,c(0.6,0.6),dt=0.001)
    xx = simTs1D(x0,0,T,0.2,stepLV1D,verb=TRUE)
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
    N=200
    T=10
    data(spnModels)
    x0=matrix(0,nrow=3,ncol=N)
    rownames(x0)=c("S","I","R")
    x0[1,]=100
    x0[2,round(N/2)]=20
    stepSIR1D = StepCLE1D(SIR,c(1,0.5,0))
    xx = simTs1D(x0,0,T,0.05,stepSIR1D,verb=TRUE)
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
