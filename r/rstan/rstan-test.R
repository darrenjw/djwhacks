## rstan-test.R
## Test of rstan...

## install.packages("rstan")

library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

stanFile = function(modelstring) {
    tmpf=tempfile(fileext="stan")
    tmps=file(tmpf, "w")
    cat(modelstring, file=tmps)
    close(tmps)
    tmpf
}

modelstring="
data {
  int<lower=1> N;
  real<lower=0,upper=1> p;
}
// no unknown model parameters
model {
}
// just use for simple forward simulation
generated quantities {
  int x[N];
  for(n in 1:N) {
    x[n] = bernoulli_rng(p);
  }
}

"

constants = list(N=30, p=0.8)
output = stan(file=stanFile(modelstring), data=constants, iter=1,
              chains=1, algorithm="Fixed_param")
out = as.matrix(output)
dim(out)
head(out)








## eof
