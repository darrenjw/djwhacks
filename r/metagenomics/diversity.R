## diversity.R

library(ebimetagenomics)

## ps = getProjectSummary("SRP047083") # Microbiome QC
## ps = getProjectSummary("ERP003634") # Tara
ps = getProjectSummary("ERP009703") # OSD

samples = projectSamples(ps)
otu = getSampleOtu(ps,samples[1])
dim(otu)
plotOtu(otu)
tad = convertOtuTad(otu)
dim(tad)
head(tad)

library(vegan)
estimateR(otu$Count)

library(breakaway)
breakaway(tad)

## simulated data
set.seed(123)
comm = rsad(S=1000,frac=0.01,sad="lnorm",coef=list(meanlog=5,sdlog=2))
length(comm)
sum(comm)

estimateR(comm)





## eof


