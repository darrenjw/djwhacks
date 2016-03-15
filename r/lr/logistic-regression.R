# logistic-regression.R
# Demonstrating logistic regression

# First simulate some data
N = 250
beta = c(0.2,0.3)
x = runif(N,-10,10)
X = cbind(rep(1,N),x)
theta = X %*% beta
expit=function(x) 1/(1+exp(-x))
p = expit(theta)
y = rbinom(N,1,p)

# Plot the data
plot(x,y,pch=19,col=2)
points(x,p,pch=19,col=3)

# Fit using logistic regression in R
mod = glm(y~x,family="binomial")
print(summary(mod))

# Now lets do it ourselves...

# first the b() function from the exp family, and it's derivatives, etc.
b=function(x) log(1+exp(x))
bp = function(x) expit(x)
bpp = function(x) {
    e = exp(-x)
    e/((1+e)*(1+e))
}

# Now IRLS...
bhat = c(0,0)
for (i in 1:10) {
    print(i)
    eta = as.vector(X %*% bhat)
    W = diag(bpp(eta))
    z = y - bp(eta)
    bhat = as.vector(bhat + solve(t(X) %*% W %*% X,t(X) %*% z))
    print(bhat)
}

# compare with R version
print(mod)
# looks good!  ;-)

# Now let's try some simple MH MCMC...
its = 20000
tune = 0.1
beta.old = bhat
post=matrix(0,nrow=its,ncol=2)
post[1,] = beta.old
oll = -1e99
for (i in 2:its) {
    message(paste(i,""),appendLF=FALSE)
    beta.prop = beta.old + rnorm(2,0,tune)
    eta = as.vector(X %*% beta.prop)
    ll = sum(eta*y) - sum(b(eta))
    if (log(runif(1,0,1)) < ll - oll) {
        beta.old = beta.prop
        oll = ll
    }
    post[i,] = beta.old
}
message("")

# Now plot the MCMC output
op=par(mfrow=c(3,3))
for (i in 1:2) {
    plot(1:its,post[,i],type="l")
    abline(h=beta[i],col=3,lwd=2)
    abline(h=bhat[i],col=4,lwd=2)
    hist(post[,i],100)
    abline(v=beta[i],col=3,lwd=2)
    abline(v=bhat[i],col=4,lwd=2)
    acf(post[,i])
}
plot(post,pch=19,col=rgb(0,0,0,0.1),main="RW MH")
points(rbind(beta,bhat),pch=19,col=3:4,cex=2)
#par(op)

# Now lets run a Langevin algorithm (raw - not MALA)
tau = 0.005
stau = sqrt(tau)
# score function
u = function(beta) {t(X) %*% (y-bp(X %*% beta))}

beta.la = bhat
postla = matrix(0,nrow=its,ncol=2)
postla[1,] = beta.la
for (i in 2:its) {
    message(paste(i,""),appendLF=FALSE)
    beta.la = beta.la + 0.5*tau*u(beta.la) + stau*rnorm(2,0,1)
    postla[i,] = beta.la
}
message("")

# Now plot the MCMC output
plot(postla,pch=19,col=rgb(0,0,0,0.1),main="(unadjusted) Langevin algorithm")
points(rbind(beta,bhat),pch=19,col=3:4,cex=2)

# Now let's run a MALA
beta.old = bhat
postmala = matrix(0,nrow=its,ncol=2)
postmala[1,] = beta.old
oll = -1e99
for (i in 2:its) {
    message(paste(i,""),appendLF=FALSE)
    beta.prop = beta.old + 0.5*tau*u(beta.old) + stau*rnorm(2,0,1)
    eta = as.vector(X %*% beta.prop)
    ll = sum(eta*y) - sum(b(eta))
    a = ll - oll +
        sum(dnorm(beta.old,beta.prop +
                           0.5*tau*u(beta.prop),rep(stau,2),log=TRUE)) -
        sum(dnorm(beta.prop,beta.old +
                            0.5*tau*u(beta.old),rep(stau,2),log=TRUE))
    if (log(runif(1,0,1)) < a) {
        beta.old = beta.prop
        oll = ll
    }    
    postmala[i,] = beta.old
}
message("")

# Now plot the MCMC output
#op=par(mfrow=c(3,3))
plot(postmala,pch=19,col=rgb(0,0,0,0.1),main="MALA")
points(rbind(beta,bhat),pch=19,col=3:4,cex=2)
par(op)




# eof

