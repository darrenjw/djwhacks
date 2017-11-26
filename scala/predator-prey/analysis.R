## analysis.R
## R code for looking at the data, etc.

df=read.csv("raw-data.csv")

## plot of the logged data
plot(df$Day,log(df$Virus),type="l",col=2,ylim=c(10,25),main="Logged data")
lines(df$Day,log(df$AOB),col=3)

## plot of the raw data
op=par(mfrow=c(2,1))
plot(df$Day,df$Virus,type="l",col=2,ylim=c(10,4.0e09),main="Virus")
plot(df$Day,df$AOB,type="l",col=3,ylim=c(10,8.0e07),main="AOB")
par(op)

## read in the MCMC output
out = read.csv("LvPmmh.csv")
#out = read.csv("LvPmmh-t15.csv.gz")
library(smfsb)
mcmcSummary(out[,c(1:4,9)])
mcmcSummary(log(out[,1:8]))
accepts = length(unique(out[,9]))
message("Acceptance rate: ",accepts/length(out[,9]))


## eof

