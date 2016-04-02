# mcmc-summ.R
# Do an mcmc summary

args = commandArgs(TRUE)
filename = "mcmc-out.csv"
if (length(args)>0)
	filename = args[1]
message(filename)
library(smfsb)
mcmcSummary(read.csv(filename))

# eof

