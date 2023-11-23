## sim.R

library(smfsb)
library(smfsbSBML)

## Modify StepGillespie to allow large hazards...
StepGillespie <- function (N) 
{
    S = t(N$Post - N$Pre)
    v = ncol(S)
    return(function(x0, t0, deltat, ...) {
        t = t0
        x = x0
        termt = t0 + deltat
        repeat {
            h = N$h(x, t, ...)
            h0 = sum(h)
            if (h0 < 1e-10) t = 1e+99 else if (h0 > 1e+15) {
                t = 1e+99
                warning("Hazard too big - terminating simulation!")
            } else t = t + rexp(1, h0)
            if (t >= termt) return(x)
            j = sample(v, 1, prob = h)
            x = x + S[, j]
        }
    })
}


system("mod2sbml.py lambda.mod > lambda.xml")

mod = sbml2spn("lambda.xml")

step = StepGillespie(mod)

out = simTs(mod$M, 0, 100, 0.1, step)
plot.ts(out, plot.type="single", col=1:ncol(out))

out = simSample(500, mod$M, 0, 60, step)
hist(out[,6])


## eof

