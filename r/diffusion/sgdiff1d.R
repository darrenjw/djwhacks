# sgdiff1d.R
# diffusion on a 1d grid with Gaussian noise

D=100 # num grid cells
S=1000 # num time steps
dc=0.1 # diffusion coefficient
dt=0.1 # time step

state=rep(0,D)
state[round(D/2)]=100*D

mat=matrix(0,nrow=S,ncol=D)

forward=function(v) c(v[2:D],v[1])
back=function(v) c(v[D],v[1:(D-1)])
laplace=function(v) forward(v) + back(v) - 2*v

for (i in 1:S) {
 noise=rnorm(D,0,sqrt(dt)) # Gaussian noise
 state=state+dc*laplace(state)*dt + sqrt(dc)*(
	sqrt(state+forward(state))*noise -
	sqrt(state+back(state))*back(noise))
 state[state<0]=-state[state<0] # reflect at zero
 mat[i,]=state
}

png("sgdiff1d.png",800,600)
image(mat[10:S,20:80],xlab="time",ylab="space",main="Continuous stochastic diffusion in 1d")
dev.off()



# eof


