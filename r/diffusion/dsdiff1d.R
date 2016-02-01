                                        # dsdiff1d.R
                                        # Discrete stochastic diffusion on a 1d grid

D=100 # num grid cells
S=30 # num time points
dt=0.05
N=S/dt

state=rep(0,D)
state[round(D/2)]=100*D

mat=matrix(0,nrow=N,ncol=D)

for (i in 1:N) {
    t=0
    repeat {
        h=c(state,state) # first left reactions and then right reactions
        h0=sum(h)
        t=t+rexp(1,h0)
        if (t>dt) break
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
    }
    mat[i,]=state
}

png("dsdiff1d.png",800,600)
image(mat[10:N,20:80],xlab="time",ylab="space",main="Discrete stochastic diffusion in 1d")
dev.off()

                                        # eof


