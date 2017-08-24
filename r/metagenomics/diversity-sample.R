## diversity-sample.R
## Generate diversity information offline for a EMG project

##########
## Assuming that a Sample->Run mapping file is available
##########

## Simple check that we are in a project directory
currentdir = tail(strsplit(getwd(),'/')[[1]],n=1)
if ((nchar(currentdir) != 9) | (substr(currentdir,3,3) != "P"))
    stop("This script must be run from a project directory")

## Drop into directory containing the run folders
setwd("version_3.0")

## Load required libraries
library(ebimetagenomics)

## Read in Sample->Run mapping file
pathbits = strsplit(getwd(),'/')[[1]]
project = pathbits[length(pathbits)-1]
mapping = read.delim(file.path("project-summary",paste(project,"txt",sep=".")),
                     as.is=TRUE)
rownames(mapping) = mapping$run_id
samples = unique(sort(mapping$sample_id))

## Analyse samples
tab = NULL
for (sample in samples) {
    message(sample)
    runs = mapping$run_id[mapping$sample_id==sample]
    rundirs = paste(runs,"FASTQ",sep="_")
    otufiles = file.path(rundirs,"cr_otus",
                         paste(rundirs,"otu_table.txt",sep="_"))
    otus = lapply(otufiles,read.otu.tsv)
    otu = Reduce(mergeOtu, otus)
    plotname = paste(paste(sample,"tad",sep="-"),"svg",sep=".")
    svg(file.path("project-summary",plotname))
    summ = analyseOtu(otu)
    dev.off()
    tab = rbind(tab,summ)
    rownames(tab)[dim(tab)[1]] = sample
}
df=data.frame("Sample"=rownames(tab),tab)
write.table(df,file.path("project-summary","diversity-sample.tsv"),
            sep="\t",row.names=FALSE)


## warnings()

## eof


