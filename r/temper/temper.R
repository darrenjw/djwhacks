# temper.R
# functions for messing around with tempering MCMC

makeU=function(gamma=1)
{
  function(x) gamma*(x*x-1)*(x*x-1)
}

U=makeU(4)

op=par(mfrow=c(2,1))
curve(U(x),-2,2,main="Potential function, U(x)")
curve(exp(-U(x)),-2,2,main="Unnormalised density function, exp(-U(x))")
par(op)

# First look at some independent chains

chain=function(target=U,iters=10000,tune=0.1,init=1)
{
  x=init
  xvec=numeric(iters)
  for (i in 1:iters) {
    can=x+rnorm(1,0,tune)
    logA=target(x)-target(can)
    if (log(runif(1))<logA)
      x=can
    xvec[i]=x
  }
  ts(xvec,start=1)
}

numChains=5
op=par(mfrow=c(numChains,2))
for (i in 1:numChains) {
  mychain=chain(makeU(i))
  plot(mychain)
  hist(mychain,50)
}

# Next, let's do 5 chains at once...

chains=function(uncurried=function(gamma,x) makeU(gamma)(x),iters=10000,tune=0.1,init=1)
{
  x=rep(init,numChains)
  xmat=matrix(0,iters,numChains)
  for (i in 1:iters) {
    can=x+rnorm(numChains,0,tune)
    logA=unlist(Map(uncurried,1:numChains,x))-unlist(Map(uncurried,1:numChains,can))
    accept=(log(runif(numChains))<logA)
    x[accept]=can[accept]
    xmat[i,]=x
  }
  xmat
}

#require(smfsb)
#mcmcSummary(chains())
mat=chains()
op=par(mfrow=c(numChains,2))
for (i in 1:numChains) {
  plot(ts(mat[,i],start=1))
  hist(mat[,i],50)
}

# Next let's couple the chains...












# eof


