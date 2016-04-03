# mcmc-summ.R
# Do an mcmc summary

args = commandArgs(TRUE)
filename = "mcmc-out.csv"
if (length(args)>0)
	filename = args[1]
message(filename)
library(smfsb)

mat=read.csv(filename)
mcmcSummary(mat)
plot(mat,pch=19,cex=0.2)

mat=log(mat)
mcmcSummary(mat)
plot(mat,pch=19,cex=0.2)


# eof

