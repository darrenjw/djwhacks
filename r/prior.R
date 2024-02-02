## prior.R
## Prior for epidemic modelling

set.seed(42)
N = 100000

gam = 1/rgamma(N, 6, 1)
bet = rgamma(N, 32, 20/gam)

hist(bet, 100, col="lightblue", xlim=c(0,1.5), freq=FALSE)

curve(exp(
    32*log(20)+lgamma(38)-lgamma(32)-lgamma(6)+31*log(x)-38*log(1+20*x)
         ), col="forestgreen", lwd=2, add=TRUE)

## eof
