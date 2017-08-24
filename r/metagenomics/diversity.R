## diversity.R
## Generate diversity information offline for a EMG project

## Simple check that we are in a project directory
currentdir = tail(strsplit(getwd(),'/')[[1]],n=1)
if ((nchar(currentdir) != 9) | (substr(currentdir,3,3) != "P"))
    stop("This script must be run from a project directory")

## Drop into directory containing the run folders
setwd("version_3.0")

## Load required libraries
library(ebimetagenomics)

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
    svg(paste(run,"svg",sep="."))
    summ = analyseOtu(otu)
    dev.off()
    tab = rbind(tab,summ)
    rownames(tab)[dim(tab)[1]] = run
    file.rename(from=paste(run,"svg",sep="."),
                to=file.path(dirname,"charts","tad-plots.svg"))
}
df=data.frame("Run"=rownames(tab),tab)
write.table(df,file.path("project-summary","diversity.tsv"),
            sep="\t",row.names=FALSE)


## eof


