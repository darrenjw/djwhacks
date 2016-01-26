# sgdiff1d.R
# diffusion on a 1d grid with Gaussian noise

D=100 # num grid cells
S=1000 # num time steps
dc=0.05 # diffusion coefficient
dt=0.1 # time step

state=rep(0,D)
state[round(D/2)]=D

mat=matrix(0,nrow=S,ncol=D)

rl=function(v) c(v[2:D],v[1])

rr=function(v) c(v[D],v[1:(D-1)])

for (i in 1:S) {
 sp=rl(state)
 sm=rr(state)
 noise=rnorm(D,0,sqrt(dt)) # Gaussian noise
 diff=sqrt(dc*(sm+state))*noise
 state=state+dc*(sp+sm-2*state)*dt # diffusion
 state=state+diff-rl(diff) # add noise but conserve mass
 state[state<0]=-state[state<0] # reflect at zero
 mat[i,]=state
}

image(mat)


# eof


