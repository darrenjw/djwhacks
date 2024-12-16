#!/usr/bin/env Rscript
## keys.R
## The random keys problem

e=exp(1)
print(e)
print(1/e)

n = 12 # number of keys

N = 1000000 # number of Monte Carlo samples
count = 0
for (i in 1:N) {
    samp = sample(n, n, replace=FALSE)
    count = count + all(!(samp == 1:n))
}
print(count/N) # empirical probability


## eof

