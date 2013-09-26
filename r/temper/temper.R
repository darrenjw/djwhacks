# temper.R
# functions for messing around with tempering MCMC

makeU=function(gamma=1)
{
  return(
      function(x) gamma*(x*x-1)*(x*x-1)
        )
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
    xvec[i]=can
  }
  ts(xvec,start=1)
}

op=par(mfrow=c(5,2))
for (i in 1:5) {
  mychain=chain(makeU(i))
  plot(mychain)
  hist(mychain,50)
}

# Next let's couple the chains...












# eof


