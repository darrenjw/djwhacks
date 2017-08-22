## diversity.R
## Generate diversity information offline for a project


##############
## TODO - remove this and run direct from relevant project directory
setwd("./zip/nfs/production/interpro/metagenomics/results/2017/05/DRP003216")
##############

## Drop into directory containing the run folders
setwd("version_3.0")

## Load required libraries
library(ebimetagenomics)
library(vegan)
library(breakaway)

## Function definitions
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

otuSummary = function(otu,label,plot=TRUE) {
    ns = dim(otu)[1]
    ni = sum(otu$Count)
    sh = diversity(otu$Count)
    fa = fisher.alpha(otu$Count)
    er = estimateR(otu$Count)
    vln = veiledspec(prestondistr(otu$Count))
    tad = convertOtuTad(otu)
    br = breakaway(tad,print=FALSE,plot=FALSE,answers=TRUE)
    mod = fitsad(otu$Count,"poilog")
    p0 = dpoilog(0,mod@coef[1],mod@coef[2])
    pln = ns/(1-p0)
    coverage = function(x){1-dpoilog(0,mod@coef[1]+log(x/ni),mod@coef[2])}
    qs = c(0.75,0.90,0.95,0.99)
    Ls = sapply(qs,function(q){intSolve(function(x){coverage(x)-q},1,10^12)})
    if (plot) {
        svg(paste(label,"svg",sep="."))
        plot(octav(otu$Count),main=paste("Preston plot for",label))
        dev.off()
    }
    c(
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
        "S.vln" = unname(vln[1]),
        "S.pln" = pln,
        "L.75" = Ls[1],
        "L.90" = Ls[2],
        "L.95" = Ls[3],
        "L.99" = Ls[4]
    )
}

## Run on a project
## Assuming current directory contains run folders...
dirlist = grep("??R??????_FASTQ",list.dirs(recursive=FALSE),value=TRUE)
tab = NULL
for (dir in dirlist) {
    dirname = strsplit(dir,.Platform$file.sep)[[1]][-1]
    run = strsplit(dirname,'_')[[1]][1]
    message(run)
    otufile = file.path(dirname,"cr_otus",
                        paste(dirname,"otu_table.txt",sep="_"))
    otu = read.otu.tsv(otufile)
    summ = otuSummary(otu,run)
    tab = rbind(tab,summ)
    rownames(tab)[dim(tab)[1]] = run
    file.rename(from=paste(run,"svg",sep="."),
                to=file.path(dirname,"charts","preston.svg"))
}
df=data.frame("Run"=rownames(tab),tab)
write.table(df,file.path("project-summary","diversity.tsv"),
            sep="\t",row.names=FALSE)




## eof


