# analysis.R
# R code for analysis of the MCMC output file, "mcmc-out.csv"


# Need the following CRAN package:
# install.packages("smfsb")

require(smfsb)

#tab=read.csv("mcmc-out.csv",colClass="numeric")
tab=read.csv("snapshot.csv",colClass="numeric")
mcmcSummary(tab[,1:8])

x=tab[,4:20]
y=tab[,21:37]

rows=sample(1:(dim(x)[1]),500)
plot(0:16,x[1,],ylim=c(0,750),type="l",lty=0,xlab="Time",ylab="Population size",main="Latent path")
for (i in rows) {
  lines(0:16,x[i,],col=rgb(0.4,0,0,0.01))
  lines(0:16,y[i,],col=rgb(0,0,0.2,0.01))
}




# eof


