# analysis.R
# R code for analysis of the MCMC output file, "mcmc-out.csv"


# Need the following CRAN package:
# install.packages("smfsb")

require(smfsb)

args <- commandArgs(trailingOnly = TRUE)
filename=args[1]
if (is.na(filename))
  filename="mcmc-out.csv"
print(filename)

tab=read.csv(filename,colClass="numeric")
mcmcSummary(tab[,1:8])

x=tab[,seq(4,34,by=2)]
y=tab[,seq(5,35,by=2)]
grid=seq(0,30,by=2)

xLower=apply(x,2,quantile,0.005)
xUpper=apply(x,2,quantile,0.995)
yLower=apply(y,2,quantile,0.005)
yUpper=apply(y,2,quantile,0.995)
plot(grid,yUpper,ylim=c(0,750),type="l", lty=0,xlab="Time",ylab="Population size",main="Latent path")
polygon(c(grid,rev(grid),grid[1]),c(yUpper,rev(yLower),yUpper[1]),col=rgb(0.5,0.5,1,0.5),border=NA)
polygon(c(grid,rev(grid),grid[1]),c(xUpper,rev(xLower),xUpper[1]),col=rgb(1,0.5,0.5,0.5),border=NA)
xLower=apply(x,2,quantile,0.25)
xUpper=apply(x,2,quantile,0.75)
yLower=apply(y,2,quantile,0.25)
yUpper=apply(y,2,quantile,0.75)
polygon(c(grid,rev(grid),grid[1]),c(yUpper,rev(yLower),yUpper[1]),col=rgb(0,0,1,0.5),border=NA)
polygon(c(grid,rev(grid),grid[1]),c(xUpper,rev(xLower),xUpper[1]),col=rgb(1,0,0,0.5),border=NA)

rows=sample(1:(dim(x)[1]),1000)
plot(grid,x[1,],ylim=c(0,750),type="l", lty=0,xlab="Time",ylab="Population size",main="Latent path")
for (i in rows) {
  lines(grid,x[i,],col=rgb(0.4,0,0,0.01))
  lines(grid,y[i,],col=rgb(0,0,0.2,0.01))
}




# eof


