## PMCMC.R

## make sure the package is loaded
require(smfsb)

require(parallel)

## code to factor out
ppfMLLik = function (n, simx0, t0, stepFun, dataLik, data) 
{
    times = c(t0, as.numeric(rownames(data)))
    deltas = diff(times)
    return(function(...) {
        xmat = simx0(n, t0, ...)
        ll = 0
        for (i in 1:length(deltas)) {
            xmat = t(parApply(cl, xmat, 1, stepFun, t0 = times[i],
                             deltat = deltas[i], ...))
            lw = parRapply(cl, xmat, dataLik, t = times[i + 1],
                           y = data[i,], log = TRUE, ...)
            m = max(lw)
            rw = lw - m
            sw = exp(rw)
            ll = ll + m + log(mean(sw))
            rows = sample(1:n, n, replace = TRUE, prob = sw)
            xmat = xmat[rows,]
        }
        ll
    })
}
## end of code to factor out


## load the reference data
data(LVdata)

## assume known measurement SD of 10
noiseSD=10

## now define the data likelihood functions
data1Lik <- function(x,t,y,log=TRUE,...)
{
	with(as.list(x),{
		return(dnorm(y,x1,noiseSD,log))
	})
}

data2Lik <- function(x,t,y,log=TRUE,...)
{
	ll=sum(dnorm(y,x,noiseSD,log=TRUE))
	if (log)
		return(ll)
	else
		return(exp(ll))
}

data3Lik <- function(x,t,y,log=TRUE,...)
{
	ll=sum(dnorm(y,x,th["sd"],log=TRUE))
	if (log)
		return(ll)
	else
		return(exp(ll))
}

## now define a sampler for the prior on the initial state
simx0 <- function(N,t0,...)
{
	mat=cbind(rpois(N,50),rpois(N,100))
	colnames(mat)=c("x1","x2")
	mat
}

LVdata=as.timedData(LVnoise10)
LVpreyData=as.timedData(LVpreyNoise10)
colnames(LVpreyData)=c("x1")

## create marginal log-likelihood functions, based on a particle filter
mLLik1=ppfMLLik(100,simx0,0,stepLVc,data1Lik,LVpreyData)
mLLik2=ppfMLLik(100,simx0,0,stepLVc,data2Lik,LVdata)
mLLik3=ppfMLLik(100,simx0,0,stepLVc,data3Lik,LVdata)

## Now create an MCMC algorithm...
th=c(th1 = 1, th2 = 0.005, th3 = 0.6)
#th=c(th1 = 1, th2 = 0.005, th3 = 0.6, sd=10)
p = length(th)
rprop = function(th, tune=0.01) { th*exp(rnorm(p,0,tune)) }

cl = makeCluster(4)
clusterExport(cl,c("noiseSD"))
thmat = metropolisHastings(th,mLLik1,rprop,iters=1000,thin=10)
stopCluster(cl)

## Dump MCMC output matrix to disk
save(thmat,file="LV.RData")

## Compute and plot some basic summaries
mcmcSummary(thmat,truth=th)

# eof

