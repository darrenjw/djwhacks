## anova.R
## Simple anova test

source("myStan.R")

## simulate some data
set.seed(1)
Z=matrix(rnorm(1000*8,3.1,0.1),nrow=8)
RE=rnorm(8,0,0.01)
X=t(Z+RE)
colnames(X)=paste("Uni",1:8,sep="")
Data=stack(data.frame(X))
boxplot(exp(values)~ind,data=Data,notch=TRUE)

stop("done")

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
output = stan(file=stanFile(modelstring), data=constants, iter=2000,
              chains=4, warmup=1000)
out = as.matrix(output)
dim(out)
head(out)

library(smfsb)
mcmcSummary(out)




## eof
