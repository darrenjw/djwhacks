# analysis.R
# R code for analysis of the MCMC output file, "mcmc-out.csv"


# Need the following CRAN package:
# install.packages("smfsb")

require(smfsb)


tab=read.csv("mcmc-out.csv",colClass="numeric")
mcmcSummary(tab)




# eof


