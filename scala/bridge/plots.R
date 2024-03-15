## plots.R

leftFP = 1
rightFP = 4


library(smfsb)

br = read.csv("bridge.csv", header=TRUE)
br = as.matrix(br)
m = dim(br)[2]+1
its = dim(br)[1]
br = cbind(rep(leftFP, its), br, rep(rightFP, its))
plot(ts(br[1,], start=0, freq=m), ylim=c(0, 5),
     col=rgb(0, 0, 1, 0.01))
for (i in 2:its)
    lines(ts(br[i,], start=0, freq=m), col=rgb(0, 0, 1, 0.01))
means = colMeans(br)
lines(ts(means, start=0, freq=m), col=2, lwd=2)

sub = br[, c(5, 25, 50, 75)]
mcmcSummary(sub)


## Unconditional simulation

mu = 1.2
sig = 0.5
uIts = 1000000
dt = 1/m
sdt = sqrt(dt)
unc = matrix(0, nrow=uIts, ncol=m+1)
unc[,1] = leftFP
for (i in 2:(m+1))
    unc[,i] = unc[,i-1] + mu*unc[,i-1]*dt +
        rnorm(uIts, 0, sig*unc[,i-1]*sdt)
plot(ts(unc[1,], start=0, freq=m), ylim=c(0, 5),
     col=rgb(0, 0, 1, 0.01))
for (i in 2:its)
    lines(ts(unc[i,], start=0, freq=m), col=rgb(0, 0, 1, 0.01))
means = colMeans(unc)
lines(ts(means, start=0, freq=m), col=2, lwd=2)

## Approximate conditional simulation

con = unc[abs(unc[,m+1]-4) < 0.02,]
plot(ts(con[1,], start=0, freq=m), ylim=c(0, 5),
     col=rgb(0, 0, 1, 0.01))
for (i in 2:dim(con)[1])
    lines(ts(con[i,], start=0, freq=m), col=rgb(0, 0, 1, 0.01))
means = colMeans(con)
lines(ts(means, start=0, freq=m), col=2, lwd=2)

sub = con[, c(5, 25, 50, 75)]
mcmcSummary(sub)


## eof

