# dsrd1d.R
# Discrete stochastic reaction diffusion on a 1d grid
# RDME Reaction diffusion master equation
# Next subvolume method

D=80 # num grid cells
T=120 # final time
dt=0.25 # time step for recording
th=c(1,0.005,0.6) # reaction rate parameters
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
    if (i<length(v)) {
     v[i+1]=v[i+1]+1
    } else {
     v[1]=v[1]+1
    }
    v
}

diffuse=function(x,y,hd) {
  r=sample(1:length(x),1,prob=hd)
  if (runif(1)<0.5) {
   if (runif(1,0,x[r]+y[r])<=x[r]) {
    x=mvleft(x,r)
   } else {
    y=mvleft(y,r)
   }
  } else {
   if (runif(1,0,x[r]+y[r])<=x[r]) {
    x=mvright(x,r)
   } else {
    y=mvright(y,r)
   }
  }
  list(x,y)
}

react=function(x,y,h,hr) {
  r=sample(1:length(x),1,prob=hr)
  u=runif(1,0,h[r,1]+h[r,2]+h[r,3])
  if (u<h[r,1]) {
   x[r]=x[r]+1
  } else if (u<h[r,1]+h[r,2]) {
   x[r]=x[r]-1
   y[r]=y[r]+1
  } else {
   y[r]=y[r]-1
  }
  list(x,y)
}

t=0
tt=dt
i=0
while (i < N) {
 # first consider reaction hazards
 h=cbind(th[1]*x,th[2]*x*y,th[3]*y)
 hr=h %*% rep(1,3) # apply(h,1,sum) is much slower...
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
  l=diffuse(x,y,hd)
  x=l[[1]]
  y=l[[2]]
 } else {
  l=react(x,y,h,hr)
  x=l[[1]]
  y=l[[2]]
 }
}


op=par(mfrow=c(1,2))
image(xmat,main="x - prey",xlab="Time",ylab="Space")
image(ymat,main="y - predator",xlab="Time",ylab="Space")
par(op)
message("Done!")



# eof


