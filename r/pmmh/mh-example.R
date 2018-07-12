## mh-example.R

library(smfsb)

## First simulate some synthetic data
data = rnorm(250,5,2)
## Now use MH to recover the parameters
llik = function(x) { sum(dnorm(data,x[1],x[2],log=TRUE)) }
prop = function(x) { rnorm(2,x,0.1) }
prior = function(x, log=TRUE) {
    l = dnorm(x[1],0,100,log=TRUE) + dgamma(x[2],1,0.0001,log=TRUE)
    if (log) l else exp(l)
}
out = metropolisHastings(c(mu=1,sig=1), llik, prop,
                         dprior=prior, verb=FALSE)
out = out[1000:10000,]
mcmcSummary(out, truth=c(5,2), rows=2, plot=FALSE)


## eof
