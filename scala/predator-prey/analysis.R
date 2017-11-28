## analysis.R
## R code for looking at the data, analysing MCMC results, etc.

args=commandArgs(trailingOnly=TRUE)
if (length(args)!=1)
    stop("Requires exactly one argument to run")
fileName=args[1]
if (!file.exists(fileName))
    stop(paste("File: ",fileName,"does not exist"))

df=read.csv("raw-data.csv")

## plot of the logged data
plot(df$Day,log(df$Virus),type="l",col=2,ylim=c(10,25),main="Logged data")
lines(df$Day,log(df$AOB),col=3)

## plot of the raw data
op=par(mfrow=c(2,1))
plot(df$Day,df$Virus,type="l",col=2,ylim=c(10,4.0e09),main="Virus")
plot(df$Day,df$AOB,type="l",col=3,ylim=c(10,8.0e07),main="AOB")
par(op)

summary(df)

## read in the MCMC output
out = read.csv(fileName)
library(smfsb)
mcmcSummary(out[,c(1:4,9)])
mcmcSummary(log(out[,6:8]))
accepts = length(unique(out[,9]))
message("Acceptance rate: ",accepts/length(out[,9]))


## eof

