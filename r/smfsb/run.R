## Run the protein model

library(smfsb)
library(libSBML)
library(smfsbSBML)

system("mod2sbml.py < protein.mod > protein.xml")
mod = sbml2spn("protein.xml")
system("mod2sbml.py < protein2.mod > protein2.xml")
mod2 = sbml2spn("protein2.xml")

pdf("protein.pdf",10,10)
op=par(mfrow=c(2,1))
out = simTs(mod$M, 0, 1000, 0.1, StepGillespie(mod))
plot(out, plot.type="single", col=2:3, ylim=c(0,400),
     main="Transcription rate: 1, RNA deg rate: 0.5")
out2 = simTs(mod$M, 0, 1000, 0.1, StepGillespie(mod2))
plot(out2, plot.type="single", col=2:3, ylim=c(0,400),
     main="Transcription rate: 10, RNA deg rate: 5")
par(op)
dev.off()

print(summary(out))
print(var(out))
print(sd(out[,2]))

## eof

