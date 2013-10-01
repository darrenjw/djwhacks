# evidence.R
# compute the log-ratio of normalising constants

U=function(gam,x)
{
  gam*(x*x-1)*(x*x-1)
}

# global settings
temps=2^(0:3)
iters=1e5

chains=function(pot=U, tune=0.1, init=1)
{
  x=rep(init,length(temps))
  xmat=matrix(0,iters,length(temps))
  for (i in 1:iters) {
    can=x+rnorm(length(temps),0,tune)
    logA=unlist(Map(pot,temps,x))-unlist(Map(pot,temps,can))
    accept=(log(runif(length(temps)))<logA)
    x[accept]=can[accept]
    swap=sample(1:length(temps),2)
    logA=pot(temps[swap[1]],x[swap[1]])+pot(temps[swap[2]],x[swap[2]])-
            pot(temps[swap[1]],x[swap[2]])-pot(temps[swap[2]],x[swap[1]])
    if (log(runif(1))<logA)
      x[swap]=rev(x[swap])
    xmat[i,]=x
  }
  colnames(xmat)=paste("gamma=",temps,sep="")
  xmat
}

mat=chains()

require(smfsb)
mcmcSummary(mat,rows=length(temps))

mat=mat[,1:(length(temps)-1)]
diffs=diff(temps)
mat=(mat*mat-1)^2
mat=-t(diffs*t(mat))
mat=exp(mat)
logEvidence=sum(log(colMeans(mat)))
message(paste("The log of the ratio of the last and first normalising constants is",logEvidence))

# eof

