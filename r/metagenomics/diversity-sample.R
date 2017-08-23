## diversity-sample.R
## Generate diversity information offline for a EMG project

##############
## TODO - remove this and run direct from relevant project directory
setwd("~/src/git/djwhacks/r/metagenomics/zip/nfs/production/interpro/metagenomics/results/2017/05/DRP003216")
##############

##########
## Assuming that a Sample->Run mapping file is available
##########

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


## warnings()

## eof


