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
            else if (h0 > 1e+06) {
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
    for (i in 2:N) {
        if (verb) cat(N-i,"")
        t = t + dt
        x = stepFun(x, t, dt, ...)
        arr[,,i] = x
    }
    if (verb) cat("\n")
    arr
}



## test...

N=40
T=50
x0=matrix(0,nrow=2,ncol=N)
rownames(x0)=c("x1","x2")
x0[,round(N/2)]=c(10,5)
data(spnModels)
stepLV1D = StepGillespie1D(LV,c(0.2,0.1))
xx = simTs1D(x0,0,T,1,stepLV1D,verb=TRUE)
op=par(mfrow=c(2,1))
image(xx[1,,])
image(xx[2,,])
par(op)

## eof

