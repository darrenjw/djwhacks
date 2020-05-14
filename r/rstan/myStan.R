## myStan.R
## my stan setup

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

## eof
