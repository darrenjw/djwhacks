## anova.R
## Simple anova test

source("myStan.R")

## simulate some data
set.seed(1)
N=1000
Z=matrix(rnorm(N*8,3.1,0.1),nrow=8)
RE=rnorm(8,0,0.01)
print(RE)
X=t(Z+RE)
colnames(X)=paste("Uni",1:8,sep="")
Data=stack(data.frame(X))
boxplot(exp(values)~ind,data=Data,notch=TRUE)

## define the stan model
modelstring="
data {
  int<lower=1> n;
  int<lower=1> p;
  real x[n,p];
}
parameters {
  real theta[p];
  real mu;
  real<lower=0> sig;
  real<lower=0> sigt;
}
model {
  for (j in 1:p) {
    theta[j] ~ normal(0, sigt);
    for (i in 1:n) {
      x[i,j] ~ normal(mu + theta[j], sig);
    }
  }
  mu ~ normal(0, 100);
  sig ~ gamma(1, 0.001);
  sigt ~ gamma(1, 0.001);
}

"

## run the stan model
constants = list(n=N, p=8, x=X)
output = stan(file=stanFile(modelstring), data=constants, iter=2000,
              chains=4, warmup=1000)
out = as.matrix(output)
dim(out)
head(out)

library(smfsb)
mcmcSummary(out)




## eof
