## sim.R

library(smfsb)
library(smfsbSBML)

## Convert model from shorthand to full SBML
system("mod2sbml.py lambda.mod > lambda.xml")

## Read SBML model and convert to an SPN
mod = sbml2spn("lambda.xml")

## Create a simulation transition kernel
step = StepGillespie(mod)

## Create and plot a single realisation of the process
print("Single realisation")
out = simTs(mod$M, 0, 100, 0.1, step)
colours = rainbow(ncol(out))
plot.ts(out, plot.type="single", lwd=2, col=colours,
        main="A lambda phage expression realisation",
        ylab="Gene expression levels (molecules)",
        xlab="Time (minutes)")
legend(0, max(out), colnames(out), lwd=2, lty=1,
       col=colours, bg="white")

## Look at the levels of CII at time 60
print("Sample at time 60")
out = simSample(250, mod$M, 0, 60, step)
cii = out[,5]
hist(cii, 30, main="CII at time 60 minutes")

## Look at average over time up to 60
print("Average up to time 60")
n = 5000
v = vector("numeric", n)
for (i in 1:n) {
    out = simTs(mod$M, 0, 60, 0.1, step)
    v[i] = mean(out[,5])
}
hist(v, 30, main="CII averaged over 60 minutes")
print(mean(v))
print(sd(v))



## eof

