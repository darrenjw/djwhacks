## runHmm.R

if (!require("pacman")) install.packages("pacman")
pacman::p_load("HiddenMarkov")

x = scan("medium.txt")
plot(ts(x))

P = rbind(c(0.99, 0.01), c(0.01, 0.99))

## Use forward algorithm to compute filtered probabilities
la = forward(x, P, c(0.5, 0.5), "norm", list(mean=c(0, 1)))
a = exp(la)
norm = apply(a, 1, sum)
fil = a/norm
print("Filtered")
print(fil)
plot(ts(fil))

## Use forward-backward algorithm to compute smoothed probabilities
lab = forwardback(x, P, c(0.5, 0.5), "norm", list(mean=c(0, 1)))
labs = lab$logalpha + lab$logbeta
mv = apply(labs, 1, max)
labadj = labs - mv
ab = exp(labadj)
norm = apply(ab, 1, sum)
smo = ab / norm
print("Smoothed")
print(smo)
plot(ts(smo))

## eof
