## diversity.R
## Generate diversity information offline for a project


##############
## TODO - remove this and run direct from relevant project directory
setwd("./zip/nfs/production/interpro/metagenomics/results/2017/05/DRP003216/")
##############

## Drop into directory containing the run folders
setwd("version_3.0")

## Load required libraries
library(ebimetagenomics)
library(vegan)
library(breakaway)

## Function definitions
otuSummary = function(otu,label,plot=TRUE) {
    ns = dim(otu)[1]
    ni = sum(otu$Count)
    sh = diversity(otu$Count)
    fa = fisher.alpha(otu$Count)
    er = estimateR(otu$Count)
    lne = veiledspec(prestondistr(otu$Count))
    tad = convertOtuTad(otu)
    br = breakaway(tad,print=FALSE,answers=TRUE)
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
        "S.ln" = unname(lne[1])
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


