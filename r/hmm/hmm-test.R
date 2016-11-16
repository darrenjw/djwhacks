## hmm-test.R

## first simulate some synthetic data
library(smfsb)
N=10000
Pi=matrix(c(0.99,0.02,0.01,0.98),nrow=2)
Pi0=rep(0.5,2)
## First simulate "by hand":
hs=rfmc(N,Pi,Pi0)
obs=hs*rnorm(N)
plot(obs)
lines(hs,col=2,lwd=2)
## now recover states
library(HiddenMarkov)
myHmm=dthmm(obs,Pi,Pi0,"norm",list(mean=c(0,0),sd=c(1,2)))
vit=Viterbi(myHmm)
lines(vit-2,col=3,lwd=2)

# Now simulate and recover states using HiddenMarkov
myHmm2=dthmm(NULL,Pi,Pi0,"norm",list(mean=c(0,0),sd=c(1,2)))
newObs=simulate(myHmm2,nsim=N)
plot(ts(newObs$x))
lines(newObs$y,col=2,lwd=2)
vit2=Viterbi(newObs)
lines(vit2-2,col=3,lwd=2)




## eof

