# sgdiff1d.R
# diffusion on a 1d grid with Gaussian noise

D=100 # num grid cells
S=1000 # num time steps
dc=0.05 # diffusion coefficient
sig=0.1 # noise strength

state=rep(0,D)
state[round(D/2)]=D

mat=matrix(0,nrow=S,ncol=D)

rl=function(v) c(v[2:D],v[1])

rr=function(v) c(v[D],v[1:(D-1)])

for (i in 1:S) {
 sp=rl(state)
 sm=rr(state)
 state=state+dc*(sp+sm-2*state) # diffusion
 noise=rnorm(D,0,sig) # Gaussian noise
 state=state+noise-rl(noise) # add noise but conserve mass
 mat[i,]=state
}

image(mat)


# eof


