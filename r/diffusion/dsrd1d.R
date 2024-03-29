# dsrd1d.R
# Discrete stochastic reaction diffusion on a 1d grid
# RDME Reaction diffusion master equation
# Next subvolume method

D=50 # num grid cells
T=100 # final time
dt=0.5 # time step for recording
th=c(1,0.005,0.6) # reaction rate parameters
dc=0.25 # diffusion coefficient - same for x and y for now

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

stepLV=function(x,y,dt) {
 t=0
 repeat {
  h=cbind(th[1]*x,th[2]*x*y,th[3]*y)
  hr=h %*% rep(1,3) # apply(h,1,sum) is much slower...
  hrs=sum(hr)
  hd=dc*(x+y)*2 # assuming common diffusion coefficient for now
  hds=sum(hd)
  h0=hrs+hds
  t=t+rexp(1,h0)
  if (t>dt)
      return(list(x,y))
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
}


# now run the simulation algorithm

for (i in 1:N) {
  message(paste(N-i," ",sep=""),appendLF=FALSE)
  l=stepLV(x,y,dt)
  x=l[[1]]
  y=l[[2]]
  xmat[i,]=x
  ymat[i,]=y
}


pdf("dsrd1d.pdf",6,4)
op=par(mfrow=c(1,2))
image(xmat,main="x - prey",xlab="Time",ylab="Space")
image(ymat,main="y - predator",xlab="Time",ylab="Space")
dev.off()
par(op)

message("Done!")



# eof


