## abc.R

library(smfsb)

## Generic code

library(parallel)
options(mc.cores=4)

abcRun <- function(n, rprior, rdist) {
    v = vector("list", n)
    p = mclapply(v, function(x){ rprior() }) # use mcMap instead??
    d = mclapply(p, rdist) # forward simulation in parallel
    pm = t(sapply(p, identity))
    if (dim(pm)[1] == 1) pm = as.vector(pm)
    dm = t(sapply(d, identity))
    if (dim(dm)[1] == 1) dm = as.vector(dm)
    list(param=pm, dist=dm)
}

abcSmc <- function(N, rprior, dprior, rdist, rperturb, dperturb, factor=10,
                   steps=15, verb=FALSE) {
    priorLW = log(rep(1/N, N))
    priorSample = mclapply(as.list(priorLW), function(x) {rprior()})
    for (i in steps:1) {
        if (verb) message(paste(i,""), appendLF=FALSE)
        out = abcSmcStep(dprior, priorSample, priorLW, rdist, rperturb,
                         dperturb, factor)
        priorSample = out[[1]]
        priorLW = out[[2]]
    }
    if (verb) message("")
    t(sapply(sample(priorSample, N, replace=TRUE, prob=exp(priorLW)), identity))
}

abcSmcStep <- function(dprior, priorSample, priorLW, rdist, rperturb,
                       dperturb, factor=10) {
    n = length(priorSample)
    mx = max(priorLW)
    rw = exp(priorLW - mx)
    prior = sample(priorSample, n*factor, replace=TRUE, prob=rw)
    prop = mcMap(rperturb, prior)
    dist = mcMap(rdist, prop) # forward simulation in parallel
    qCut = quantile(unlist(dist), 1/factor)
    new = prop[dist < qCut]
    lw = mcMap( function(th) {
        terms = priorLW + sapply(priorSample,
                                 function(x){dperturb(x, th, log=TRUE)})
        mt = max(terms)
        denom = mt + log(sum(exp(terms - mt)))
        dprior(th, log=TRUE) - denom
    } , new)
    lw = unlist(lw)
    mx = max(lw)
    lw = lw - mx
    nlw = log(exp(lw)/sum(exp(lw)))
    list(sample = new, lw = nlw)
}



## Example code

data(LVdata)

rprior <- function() { exp(runif(3, -5, 3)) }
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
hist(log(accepted[,1]))
hist(log(accepted[,2]))
hist(log(accepted[,3]))
par(op)
pairs(log(accepted))


cat("Better summary stats\n")
ss1d <- function(vec)
{
    acs=as.vector(acf(vec, lag.max=3, plot=FALSE)$acf)[2:4]
    c(mean(vec), log(var(vec)+1), acs)
}

ssi <- function(ts)
{
    c(ss1d(ts[,1]), ss1d(ts[,2]), cor(ts[,1],ts[,2]))
}

cat("Pilot run\n")
out = abcRun(10000, rprior, function(th) { ssi(rmodel(th)) })
sds = apply(out$dist, 2, sd)

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
hist(log(accepted[,1]))
hist(log(accepted[,2]))
hist(log(accepted[,3]))
par(op)
pairs(log(accepted))


cat("Now ABC-SMC\n")
## Do prior on log space now...
rprior <- function() { runif(3, -5, 3) }
dprior <- function(x, ...) { sum(dunif(x, -5, 3, ...)) }
rmodel <- function(th) { simTs(c(50,100), 0, 30, 2, stepLVc, exp(th)) }
rperturb <- function(th){th + rnorm(3, 0, 0.5)}
dperturb <- function(thOld, thNew, ...){sum(dnorm(thNew, thOld, 0.5, ...))}

out = abcSmc(10000, rprior, dprior, rdist, rperturb,
             dperturb, verb=TRUE, steps=8)

print(summary(out))

op=par(mfrow=c(3,1))
hist(out[,1])
hist(out[,2])
hist(out[,3])
par(op)
pairs(out)

## eof

