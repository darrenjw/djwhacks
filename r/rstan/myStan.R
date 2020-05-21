## myStan.R
## my stan setup

library(rstan)

rstan_options(auto_write = TRUE)

options(mc.cores = parallel::detectCores())


## eof
