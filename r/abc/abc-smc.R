## abc-smc.R

library(smfsb)
options(mc.cores=4)
data(LVdata)
distance <- function(s) {
    diff = s - ssd
    sqrt(sum(diff*diff))
}
ss1d <- function(vec) {
    acs=as.vector(acf(vec, lag.max=3, plot=FALSE)$acf)[2:4]
    c(mean(vec), log(var(vec)+1), acs)
}
ssi <- function(ts) {
    c(ss1d(ts[,1]), ss1d(ts[,2]), cor(ts[,1],ts[,2]))
}
cat("Pilot run\n")
rprior <- function() { c(runif(1, -3, 3), runif(1, -8, -2), runif(1, -4, 2)) }
dprior <- function(x, ...) { dunif(x[1], -3, 3, ...) + 
		dunif(x[2], -8, -2, ...) + dunif(x[3], -4, 2, ...) }
rmodel <- function(th) { simTs(c(50,100), 0, 30, 2, stepLVc, exp(th)) }
out = abcRun(100000, rprior, function(th) { ssi(rmodel(th)) })
sds = apply(out$dist, 2, sd)
print(sds)
sumStats <- function(ts) { ssi(ts)/sds }
ssd = sumStats(LVperfect)
rdist <- function(th) { distance(sumStats(rmodel(th))) }

cat("Now ABC-SMC\n")
rperturb <- function(th){th + rnorm(3, 0, 0.5)}
dperturb <- function(thNew, thOld, ...){sum(dnorm(thNew, thOld, 0.5, ...))}
out = abcSmc(2000, rprior, dprior, rdist, rperturb,
             dperturb, verb=TRUE, steps=8, factor=5)
print(summary(out))

op=par(mfrow=c(3,1))
hist(out[,1],xlim=c(-3,3))
abline(v=log(1),col=2,lwd=2)
hist(out[,2],xlim=c(-8,-2))
abline(v=log(0.005),col=2,lwd=2)
hist(out[,3],xlim=c(-4,2))
abline(v=log(0.6),col=2,lwd=2)
par(op)
pairs(out)

## eof

