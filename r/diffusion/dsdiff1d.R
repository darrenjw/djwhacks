# dsdiff1d.R
# Discrete stochastic diffusion on a 1d grid

D=100 # num grid cells
S=10000 # num reaction events

state=rep(0,D)
state[round(D/2)]=D

mat=matrix(0,nrow=S,ncol=D)

for (i in 1:S) {
 h=c(state,state) # first left reactions and then right reactions
 # h0=sum(h)
 r=sample(1:(2*D),1,prob=h)
 if (r<=D) {
  # left
  state[r]=state[r]-1
  state[r-1]=state[r-1]+1
 }
 else {
  # right
  r=r-D
  state[r]=state[r]-1
  state[r+1]=state[r+1]+1
 }
 mat[i,]=state
}

image(mat)

# eof


