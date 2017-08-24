## comparisons-sample.R

##########
## Assuming that a Sample->Run mapping file is available
##########

## Load required BIOCONDUCTOR libraries
library(phyloseq)
library(DESeq2)

## Simple check that we are in a project directory
currentdir = tail(strsplit(getwd(),'/')[[1]],n=1)
if ((nchar(currentdir) != 9) | (substr(currentdir,3,3) != "P"))
    stop("This script must be run from a project directory")

## Drop into the directory containing the runs folders
setwd("version_3.0")

## Read in Sample->Run mapping file
pathbits = strsplit(getwd(),'/')[[1]]
project = pathbits[length(pathbits)-1]
mapping = read.delim(file.path("project-summary",paste(project,"txt",sep=".")),
                     as.is=TRUE)
rownames(mapping) = mapping$run_id
samples = unique(sort(mapping$sample_id))

## Read EBI phylum table
kptab = read.table("project-summary/phylum_taxonomy_abundances_v3.0.tsv",as.is=TRUE,header=TRUE)
rownames(kptab) = paste(kptab[,1],kptab[,2],sep=";")
kptab = kptab[,-(1:2)]

## PCA
colsums = apply(kptab,2,sum)
normtab = sweep(kptab,2,colsums,"/")
pca = prcomp(t(normtab))
svg("project-summary/pca-sample.svg")
colours=rainbow(length(samples))
plot(pca$x[,1],pca$x[,2],pch=19,
     col=colours[factor(mapping[rownames(pca$x),"sample_id"])],
     xlab="PCA1",ylab="PCA2",
     main="PCA for runs (based on phylum proportions)")
text(pca$x[,1],pca$x[,2],rownames(pca$x),pos=3,cex=0.6)
dev.off()

## Phyloseq analysis for differential expression...
kpps = otu_table(kptab,taxa_are_rows=TRUE)
sdtab = data.frame(Run=sample_names(kpps),
                   Sample=mapping[sample_names(kpps),"sample_id"],
                   row.names=sample_names(kpps),
                   stringsAsFactors=FALSE)
sdps = sample_data(sdtab)
pseq = phyloseq(kpps,sdps)
dedat = phyloseq_to_deseq2(pseq,~Sample)
defit = DESeq(dedat, test="Wald", fitType="parametric")
fchm <- function(base,all=samples){
    comp = all[all != base]
    mat = sapply(comp,
                 function(x){
                     results(defit,c("Sample",base,x),cooksCutoff=FALSE)$log2FoldChange
                 })
    rownames(mat) = rownames(kpps)
    mat = cbind(rep(1,dim(mat)[1]),mat)
    colnames(mat)[1] = base
    svg(file.path("project-summary",paste(base,"svg",sep=".")))
    heatmap(mat,scale="none", main=paste("Fold-change compared to", base))
    dev.off()
    invisible(mat)
}
lapply(samples,fchm)


## eof
