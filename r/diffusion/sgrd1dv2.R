# sgrd1d.R
# reaction diffusion on a 1d grid with Gaussian noise

D=100 # num grid cells
T=100 # finish time
dc=1.0 # diffusion coefficient
dt=0.2 # time step
th=c(1,0.01,0.6) # reaction rate parameters

Sto=matrix(c(1,-1,0,0,1,-1),ncol=3,byrow=TRUE)

S=T/dt

x=rep(0,D)
x[round(D/2)]=60
y=rep(0,D)
y[round(D/2)]=20

xmat=matrix(0,nrow=S,ncol=D)
ymat=matrix(0,nrow=S,ncol=D)

forward=function(v) c(v[2:D],v[1])
back=function(v) c(v[D],v[1:(D-1)])
laplace=function(v) forward(v) + back(v) - 2*v

rectify=function(v) {
 # v[v<0]=-v[v<0] # reflect at zero
 v[v<0]=0 # absorb at zero
 v
}

diffuse=function(v) {
    noise=rnorm(D,0,sqrt(dt)) # Gaussian noise
    v=v+dc*laplace(v)*dt + sqrt(dc)*(
        sqrt(v+forward(v))*noise -
        sqrt(v+back(v))*back(noise))
    v=rectify(v)
    v
}

for (i in 1:S) {
 # first diffuse
 x=diffuse(x)
 y=diffuse(y)
 # next react
 h=cbind(th[1]*x,th[2]*x*y,th[3]*y)
 dwt=matrix(rnorm(3*D,0,sqrt(dt)),ncol=3)
 x = rectify(x + (h*dt + sqrt(h)*dwt) %*% Sto[1,])
 y = rectify(y + (h*dt + sqrt(h)*dwt) %*% Sto[2,])
 # store results
 xmat[i,]=x
 ymat[i,]=y
}

pdf("sgrd1dv2.pdf",6,4)
op=par(mfrow=c(1,2))
image(xmat,main="x - prey",xlab="Time",ylab="Space")
image(ymat,main="y - predator",xlab="Time",ylab="Space")
dev.off()
par(op)


# eof


