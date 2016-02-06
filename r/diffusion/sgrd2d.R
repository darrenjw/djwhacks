# sgrd2d.R
# reaction diffusion on a 2d grid with Gaussian noise

D=200 # num grid cells
T=250 # finish time
dc=0.5 # diffusion coefficient
dt=0.2 # time step
th=c(1,0.05,0.6) # reaction rate parameters

Sto=matrix(c(1,-1,0,0,1,-1),ncol=3,byrow=TRUE)

S=T/dt

x=matrix(0,ncol=D,nrow=D)
x[round(D/2),round(D/2)]=60
y=matrix(0,ncol=D,nrow=D)
y[round(D/2),round(D/2)]=20

up=function(m) rbind(m[2:D,],m[1,])
down=function(m) rbind(m[D,],m[1:(D-1),])
left=function(m) cbind(m[,2:D],m[,1])
right=function(m) cbind(m[,D],m[,1:(D-1)])
laplace=function(m) up(m)+down(m)+left(m)+right(m)-4*m

rectify=function(m) {
    # m[m<0]=-m[m<0] # reflect at zero
    m[m<0]=0 # absorb at zero
    m
}

diffuse=function(m) {
    dwt=matrix(rnorm(D*D,0,sqrt(dt)),ncol=D,nrow=D)
    dwts=matrix(rnorm(D*D,0,sqrt(dt)),ncol=D,nrow=D)
    m = m + dc*laplace(m)*dt + sqrt(dc)*(
        sqrt(m+left(m))*dwt - sqrt(m+right(m))*right(dwt)
        + sqrt(m+up(m))*dwts - sqrt(m+down(m))*down(dwts)
    )
    #print(paste("min is",min(m)))
    m=rectify(m)
    m
}

react=function(x,y) {
 h1=th[1]*x
 h2=th[2]*x*y
 h3=th[3]*y
 dw1t=matrix(rnorm(D*D,0,sqrt(dt)),ncol=D,nrow=D)
 dw2t=matrix(rnorm(D*D,0,sqrt(dt)),ncol=D,nrow=D)
 dw3t=matrix(rnorm(D*D,0,sqrt(dt)),ncol=D,nrow=D)
 x = rectify(x + Sto[1,1]*(h1*dt+sqrt(h1)*dw1t)
    +Sto[1,2]*(h2*dt+sqrt(h2)*dw2t)
    +Sto[1,3]*(h3*dt+sqrt(h3)*dw3t))
 y = rectify(y + Sto[2,1]*(h1*dt+sqrt(h1)*dw1t)
    +Sto[2,2]*(h2*dt+sqrt(h2)*dw2t)
    +Sto[2,3]*(h3*dt+sqrt(h3)*dw3t))
 list(x,y)
}

op=par(mfrow=c(1,2))

for (i in 1:S) {
 # first diffuse
 x=diffuse(x)
 y=diffuse(y)
 # next react
 xy=react(x,y)
 x=xy[[1]]
 y=xy[[2]]
 # plot results
 message(paste(i,""),appendLF=FALSE)
 #print(paste(i,": max x is",max(x),"and max y is",max(y)))
 #print(paste(i,": sum x is",sum(x),"and sum y is",sum(y)))
 image(x,main="x - prey",xlab="Time",ylab="Space")
 image(y,main="y - predator",xlab="Time",ylab="Space")
}

par(op)


# eof


