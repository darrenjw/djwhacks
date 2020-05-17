## lr-test.R
## Simple linear regression test

source("myStan.R")

## simulate some data
n = 100
x = rnorm(n, 10, 5)
y = 1.5*x + rnorm(n, 3, 4) # alpha=3, beta=1.5, sig=4

## define the stan model
modelstring="
data {
  int<lower=1> N;
  real y[N];
  real x[N];
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sig;
}
model {
  for (i in 1:N) {
    y[i] ~ normal(alpha + beta*x[i], sig);
  }
  alpha ~ normal(0, 100);
  beta ~ normal(0, 100);
  sig ~ gamma(1, 0.01);
}

"

## run the stan model
constants = list(N=n, x=x, y=y)
output = stan(model_code=modelstring, data=constants, iter=2000,
              chains=4, warmup=1000)
out = as.matrix(output)
dim(out)
head(out)

library(smfsb)
mcmcSummary(out)




## eof
