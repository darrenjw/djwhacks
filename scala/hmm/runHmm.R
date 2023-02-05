## runHmm.R

if (!require("pacman")) install.packages("pacman")
pacman::p_load("HiddenMarkov")

x = scan("short.txt")
plot(ts(x))

P = rbind(c(0.9, 0.1), c(0.1, 0.9))

## Use forward algorithm to compute filtered probabilities
la = forward(x, P, c(0.5,0.5), "norm", list(mean=c(0, 3)))
a = exp(la)
norm = apply(a, 1, sum)
fil = a/norm
print(fil)
plot(ts(fil))

## eof
