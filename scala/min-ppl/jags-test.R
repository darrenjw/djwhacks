## jags-test.R
## Test by comparing against JAGS

## Normal random sample
require(rjags)
x=c(8,9,7,7,8,10)
data=list(x=x,n=length(x))
init=list(tau=1,mu=0)
modelstring="
  model {
    for (i in 1:n) {
      x[i]~dnorm(mu,tau)
    }
    mu~dnorm(0,0.01)
    tau~dgamma(1,0.1)
  }
"
model=jags.model(textConnection(modelstring),
                data=data,inits=init)
update(model,n.iter=100)
output=coda.samples(model=model,variable.names=c("mu","tau"),
            n.iter=10000,thin=1)
print(summary(output))
plot(output)


## Noisy count
require(rjags)
x=c(4.2,5.1,4.6,3.3,4.7,5.3)
data=list(x=x,n=length(x))
init=list(tau=1,count=2)
modelstring="
  model {
    for (i in 1:n) {
      x[i]~dnorm(count,tau)
    }
    count~dpois(10)
    tau~dgamma(1,0.1)
  }
"
model=jags.model(textConnection(modelstring),
                data=data,inits=init)
update(model,n.iter=100)
output=coda.samples(model=model,variable.names=c("count","tau"),
            n.iter=10000,thin=1)
print(summary(output))
plot(output)




## eof

