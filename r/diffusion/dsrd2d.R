# dsrd2d.R
# Discrete stochastic reaction diffusion on a 2d grid
# RDME Reaction diffusion master equation
# Next subvolume method

D=50 # num grid cells (DxD grid)
T=30 # final time
dt=0.25 # time step for recording
th=c(1,0.005,0.6) # reaction rate parameters
dc=0.5 # diffusion coefficient - same for x and y for now

N=T/dt
x=matrix(0,ncol=D,nrow=D)
x[round(D/2),round(D/2)]=60
y=matrix(0,ncol=D,nrow=D)
y[round(D/2),round(D/2)]=20

mvleft=function(m,i,j) {
    m[i,j]=m[i,j]-1
    if (j>1) {
     m[i,j-1]=m[i,j-1]+1
    } else {
     l=dim(m)[2]
     m[i,l]=m[i,l]+1
    }
    m
}

mvright=function(m,i,j) {
    m[i,j]=m[i,j]-1
    if (j<dim(m)[2]) {
     m[i,j+1]=m[i,j+1]+1
    } else {
     m[i,1]=m[i,1]+1
    }
    m
}

mvup=function(m,i,j) {
    m[i,j]=m[i,j]-1
    if (i>1) {
     m[i-1,j]=m[i-1,j]+1
    } else {
     l=dim(m)[1]
     m[l,j]=m[l,j]+1
    }
    m
}

mvdown=function(m,i,j) {
    m[i,j]=m[i,j]-1
    if (i<dim(m)[1]) {
     m[i+1,j]=m[i+1,j]+1
    } else {
     m[1,j]=m[1,j]+1
    }
    m
}


diffuse=function(x,y,hd) {
  r=sample(0:(length(as.vector(x))-1),1,prob=as.vector(hd))
  i=1 + r %% D
  j=1 + r %/% D
  u=runif(1)
  if (u<0.25) {
   if (runif(1,0,x[i,j]+y[i,j])<=x[i,j]) {
    x=mvleft(x,i,j)
   } else {
    y=mvleft(y,i,j)
   }
  } else if (u<0.5) {
   if (runif(1,0,x[i,j]+y[i,j])<=x[i,j]) {
    x=mvright(x,i,j)
   } else {
    y=mvright(y,i,j)
   }
  } else if (u<0.75) {
   if (runif(1,0,x[i,j]+y[i,j])<=x[i,j]) {
    x=mvup(x,i,j)
   } else {
    y=mvup(y,i,j)
   }    
  } else {
   if (runif(1,0,x[i,j]+y[i,j])<=x[i,j]) {
    x=mvdown(x,i,j)
   } else {
    y=mvdown(y,i,j)
   }  
  }
  # if ((min(x)<0)|(min(y)<0)) stop("negative species count after diffusion step")
  list(x,y)
}

react=function(x,y,h,hr) {
  r=sample(0:(length(as.vector(x))-1),1,prob=as.vector(hr))
  i=1 + r %% D
  j=1 + r %/% D  
  u=runif(1,0,h[[1]][i,j]+h[[2]][i,j]+h[[3]][i,j])
  if (u<h[[1]][i,j]) {
   x[i,j]=x[i,j]+1
  } else if (u<h[[1]][i,j]+h[[2]][i,j]) {
   x[i,j]=x[i,j]-1
   y[i,j]=y[i,j]+1
  } else {
   y[i,j]=y[i,j]-1
  }
  # if ((min(x)<0)|(min(y)<0)) stop("negative species count after reaction step")
  list(x,y)
}

stepLV=function(x,y,dt) {
 t=0
 repeat {
  #print(t)
  #print(x)
  #print(y)
  h=list(th[1]*x,th[2]*x*y,th[3]*y)
  hr=h[[1]]+h[[2]]+h[[3]]
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

op=par(mfrow=c(1,2))
for (i in 1:N) {
  message(paste(N-i," ",sep=""),appendLF=FALSE)
  l=stepLV(x,y,dt)
  x=l[[1]]
  y=l[[2]]
  image(x,main="x - prey")
  image(y,main="y - predator")
}


par(op)
message("Done!")



# eof


