## coverage.R

library(ebimetagenomics)
library(vegan)
library(breakaway)

otuSummary = function(otu,label,plot=TRUE) {
    ns = dim(otu)[1]
    ni = sum(otu$Count)
    sh = diversity(otu$Count)
    fa = fisher.alpha(otu$Count)
    er = estimateR(otu$Count)
    lne = veiledspec(prestondistr(otu$Count))
    tad = convertOtuTad(otu)
    br = breakaway(tad,print=FALSE,plot=FALSE,answers=TRUE)
    if (plot) {
        svg(paste(label,"svg",sep="."))
        plot(octav(otu$Count),main=paste("Preston plot for",label))
        dev.off()
    }
    c(
        "id" = label,
        "S.obs" = ns,
        "N.obs" = ni,
        "Shannon.index" = sh,
        "Fisher.alpha" = fa,
        er["S.chao1"],
        er["se.chao1"],
        er["S.ACE"],
        er["se.ACE"],
        "S.break" = br$est,
        "se.break" = br$se,
        "S.ln" = lne[1]
    )
}

## Run on a project

## ps = getProjectSummary("SRP047083") # Microbiome QC
## ps = getProjectSummary("ERP003634") # Tara
ps = getProjectSummary("ERP009703") # OSD

## Iterate over samples/runs

samples = projectSamples(ps)
otu = getSampleOtu(ps,samples[1])
otuSummary(otu,samples[1])

renyi(otu$Count)


## simulated data
set.seed(123)
comm = rsad(S=1000,frac=0.01,sad="lnorm",coef=list(meanlog=5,sdlog=2))
length(comm)
sum(comm)

otuSummary(data.frame(Count=comm),"Simulated")
diversity(comm)

veiledspec(prestondistr(comm))

mod = fitsad(comm,"poilog")
summary(mod)
op=par(mfrow=c(2,2))
plot(mod)
par(op)
mod@coef
p0=dpoilog(0,mod@coef[1],mod@coef[2])
length(comm)/(1-p0)
coverage = function(x){1-dpoilog(0,mod@coef[1]+log(x/sum(comm)),mod@coef[2])}
Ns=10^(2:12)
Ns
sapply(Ns,coverage)

intSolve=function(f,l,h){
    if (abs(l-h) < 2) {
        h
    } else {
        m = round((l+h)/2)
        if (f(m) < 0)
            intSolve(f,m,h)
        else
            intSolve(f,l,m)
    }
}

intSolve(function(x){coverage(x)-0.95},1,10^12)

qs=c(0.75,0.90,0.95,0.99)
sapply(qs,function(q){intSolve(function(x){coverage(x)-q},1,10^12)})


## eof


