## coverage.R

library(ebimetagenomics)

## Run on a project

## ps = getProjectSummary("SRP047083") # Microbiome QC
## ps = getProjectSummary("ERP003634") # Tara
ps = getProjectSummary("ERP009703") # OSD

## Iterate over samples/runs

samples = projectSamples(ps)
otu = getSampleOtu(ps,samples[1])
analyseOtu(otu)

renyi(otu$Count)

## simulated data
set.seed(123)
comm = rsad(S=1000,frac=0.01,sad="lnorm",coef=list(meanlog=4,sdlog=2))
length(comm)
sum(comm)

analyseOtu(data.frame(Count=comm))

diversity(comm)




## eof


