## abc.R

library(smfsb)

## Generic code







## Example code

data(LVdata)

rprior = function() { exp(runif(3, -4, 2)) }

rmodel = function(th) { simTs(rpois(2,50), 0, 30, 2, stepLVc, th) }
## compare to LVperfect...
sumStats = identity
ssd = sumStats(LVperfect)
distance = function(s) {
    diff = s - ssd
    sqrt(sum(diff*diff))
}


## eof

