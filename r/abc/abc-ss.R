## abc-ss.R

library(smfsb)
options(mc.cores=4)
data(LVdata)

rprior <- function() { exp(c(runif(1, -3, 3),runif(1,-8,-2),runif(1,-4,2))) }
rmodel <- function(th) { simTs(c(50,100), 0, 30, 2, stepLVc, th) }
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
out = abcRun(100000, rprior, function(th) { ssi(rmodel(th)) })
sds = apply(out$dist, 2, sd)
print(sds)
cat("Main run with calibrated summary stats\n")
sumStats <- function(ts) { ssi(ts)/sds }
ssd = sumStats(LVperfect)
rdist <- function(th) { distance(sumStats(rmodel(th))) }
out = abcRun(1000000, rprior, rdist)
q=quantile(out$dist, c(0.01, 0.05, 0.1))
print(q)
accepted = out$param[out$dist < q[1],]
print(summary(accepted))
print(summary(log(accepted)))

op=par(mfrow=c(3,1))
hist(log(accepted[,1]),xlim=c(-3,3))
abline(v=log(1),col=2,lwd=2)
hist(log(accepted[,2]),xlim=c(-8,-2))
abline(v=log(0.005),col=2,lwd=2)
hist(log(accepted[,3]),xlim=c(-4,2))
abline(v=log(0.6),col=2,lwd=2)
par(op)
pairs(log(accepted))



## eof

