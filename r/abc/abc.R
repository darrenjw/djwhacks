## abc.R

library(smfsb)
#options(mc.cores=4)
data(LVdata)

rprior <- function() { exp(c(runif(1, -3, 3),runif(1,-8,-2),runif(1,-4,2))) }
rmodel <- function(th) { simTs(c(50,100), 0, 30, 2, stepLVc, th) }
## Compare to LVperfect...
sumStats <- identity
ssd = sumStats(LVperfect)
distance <- function(s) {
    diff = s - ssd
    sqrt(sum(diff*diff))
}
rdist <- function(th) { distance(sumStats(rmodel(th))) }

cat("Simple rejection sampler\n")
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

