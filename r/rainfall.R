## rainfall.R
## Think about rainfall at a single site

n = 1000
theta = rep(0.0, n)
S = rep(0.0, n)
for (i in 2:n) {
    S[i] = 0.98*S[i-1] + rnorm(1, 0, 0.2)
    theta[i] = 0.99*theta[i-1] + S[i] + rnorm(1)
    }
yr = theta + 4 + rnorm(n,0,0.1)
yg = theta + 2 + rnorm(n,0,0.4)
yro = yr
yro[yro < 0] = 0
ygo = yg
ygo[ygo < 0] = 0

plot(ts(yro), col=2)
lines(ts(ygo), col=3)


## eof

