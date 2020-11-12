## Run the protein model

library(smfsb)
library(libSBML)
library(smfsbSBML)

system("mod2sbml.py < protein.mod > protein.xml")

mod = sbml2spn("protein.xml")
step = StepGillespie(mod)

out = simTs(mod$M, 0, 1000, 0.1, step)
plot(out)

print(summary(out))
print(var(out))
print(sd(out[,2]))

## eof

