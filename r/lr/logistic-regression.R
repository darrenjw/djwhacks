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
bp = function(x) expit(x)
bpp = function(x) {
    e = exp(-x)
    e/((1+e)*(1+e))
}
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

# Now let's try some MCMC...
its = 20000
tune = 0.1
beta.old = bhat
post=matrix(0,nrow=its,ncol=2)
post[1,] = beta.old
oll = -1e99
b=function(x) log(1+exp(x))
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
plot(post,pch=19,col=rgb(0,0,0,0.1))
points(rbind(beta,bhat),pch=19,col=3:4,cex=2)
par(op)



# eof

