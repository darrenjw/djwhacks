# dsrd1d.R
# Discrete stochastic reaction diffusion on a 1d grid
# RDME Reaction diffusion master equation
# Next subvolume method

D=50 # num grid cells
T=80 # final time
dt=0.2 # time step for recording

th1=1
th2=0.005
th3=0.6

dc=0.1 # diffusion coefficient - same for x and y for now

N=T/dt

x=rep(0,D)
x[round(D/2)]=60

y=rep(0,D)
y[round(D/2)]=20

xmat=matrix(0,nrow=N,ncol=D)
ymat=matrix(0,nrow=N,ncol=D)

mvleft=function(v,i) {
    v[i]=v[i]-1
    if (i>1) {
     v[i-1]=v[i-1]+1
    } else {
     l=length(v)
     v[l]=v[l]+1
    }
    v
}

mvright=function(v,i) {
    v[i]=v[i]-1
    if (r<length(v)) {
     v[i+1]=v[i+1]+1
    } else {
     v[1]=v[1]+1
    }
    v
}

t=0
tt=dt
i=0
while (i < N) {
 # first consider reaction hazards
 h1=th1*x
 h2=th2*x*y
 h3=th3*y
 hr=h1+h2+h3
 hrs=sum(hr)
 hd=dc*(x+y)*2 # assuming common diffusion coefficient for now
 hds=sum(hd)
 h0=hrs+hds
 t=t+rexp(1,h0)
 if (t>tt) {
     i=i+1
     tt=tt+dt
     xmat[i,]=x
     ymat[i,]=y
     message(paste(N-i," ",sep=""),appendLF=FALSE)
 }
 if (runif(1,0,h0)<hds) {
  # diffuse
  r=sample(1:(2*D),1,prob=c(hd,hd))
  if (r<=D) {
   # left
   if (runif(1,0,x[r]+y[r])<=x[r]) {
    x=mvleft(x,r)
   } else {
    y=mvleft(y,r)
   }
  } else {
   # right
   r=r-D
   if (runif(1,0,x[r]+y[r])<=x[r]) {
    x=mvright(x,r)
   } else {
    y=mvright(y,r)
   }
  }
 } else {
  # react
  r=sample(1:D,1,prob=hr)
  u=runif(1,0,h1[r]+h2[r]+h3[r])
  if (u<h1[r]) {
   x[r]=x[r]+1
  } else if (u<h1[r]+h2[r]) {
   x[r]=x[r]-1
   y[r]=y[r]+1
  } else {
   y[r]=y[r]-1
  }
 }
}


op=par(mfrow=c(1,2))
image(xmat,main="x - prey",xlab="Time",ylab="Space")
image(ymat,main="y - predator",xlab="Time",ylab="Space")
#image(xmat[round(S/10):S,])
#image(ymat[round(S/10):S,])
par(op)


# eof


