## plots.R

library(smfsb)

br = read.csv("bridge.csv", header=TRUE)
br = as.matrix(br)
m = dim(br)[2]+1
its = dim(br)[1]
plot(ts(br[1,],start=1/m,freq=m), ylim=c(0,5), col=rgb(0,0,1,0.01))
for (i in 2:its) {
    lines(ts(br[i,],start=1/m,freq=m), col=rgb(0,0,1,0.01))
}
means = colMeans(br)
lines(ts(means,start=1/m,freq=m), col=2, lwd=2)

sub = br[, c(5, 25, 50, 75)]
mcmcSummary(sub)

## eof

