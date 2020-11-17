## Run the protein model
## Directly read the SBML model "protein.xml", bypassing the shorthand stage

library(smfsb)
library(libSBML)
library(smfsbSBML)

mod = sbml2spn("protein.xml")

out = simTs(mod$M, 0, 1000, 0.1, StepGillespie(mod))
plot(out, plot.type="single", col=2:3, ylim=c(0,400),
     main="Transcription rate: 1, RNA deg rate: 0.5")

print(summary(out))
print(var(out))
print(sd(out[,2]))

## eof

