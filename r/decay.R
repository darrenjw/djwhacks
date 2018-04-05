## decay.R

IN = 1000000
N = IN
p = 0.001
n = 4000
v = vector("numeric",n)
for (i in 1:n) {
    decays = rbinom(1,N,p)
    N = N - decays
    v[i]=N
}
plot(ts(v),ylim=c(0,IN))



## eof


