## logreg-test.R
## Simple logistic regression test

source("myStan.R")

## simulate some data
set.seed(1)
n = 5000
x = rnorm(n, 1, 3)
eta = 0.4*x + 0.2 # alpha=0.2, beta=0.4
p = 1 / (exp(-eta) + 1)
y = rbinom(n, 1, p)
print(y[1:30])

## define the stan model
modelstring="
data {
  int<lower=1> N;
  int<lower=0, upper=1> y[N];
  real x[N];
}
parameters {
  real alpha;
  real<lower=0> beta;
}
model {
  for (i in 1:N) {
    real eta = alpha * beta*x[i];
    real p = 1/(1+exp(-eta));
    y[i] ~ binomial(1, p);
  }
  alpha ~ normal(0, 1);
  beta ~ lognormal(0, 2);
}

"

## run the stan model
constants = list(N=n, x=x, y=y)
<<<<<<< HEAD
output = stan(file=stanFile(modelstring), data=constants, iter=1000000,
              chains=4, warmup=10000, thin=250)
=======
output = stan(model_code=modelstring, data=constants, iter=10000,
              chains=4, warmup=2000, thin=4)
>>>>>>> 7ab72dac93751e4d62efbc63f1fae808167960a5
out = as.matrix(output)
dim(out)
head(out)

library(smfsb)
mcmcSummary(out)




## eof
