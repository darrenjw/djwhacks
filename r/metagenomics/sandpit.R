## sandpit.R

## This is just a junk file for me to try things out...

library(ebimetagenomics)

## ps = getProjectSummary("SRP047083") # Microbiome QC
## ps = getProjectSummary("ERP003634") # Tara
ps = getProjectSummary("ERP009703") # OSD

## Iterate over samples/runs
samples = projectSamples(ps)
otu1 = getSampleOtu(ps,samples[1])
analyseOtu(otu1)
otu2 = getSampleOtu(ps,samples[2])
analyseOtu(otu2)
## renyi(otu$Count)

## Convert OTU to phyloseq format...
library(phyloseq)
convertOtuPhyloseq = function(otu,sampleName="Sample"){
    m = as.matrix(otu$Count)
    rownames(m) = otu$OTU
    colnames(m) = sampleName
    otu_table(m,taxa_are_rows=TRUE)
    }

potu1=convertOtuPhyloseq(otu1,samples[1])
head(potu1)
head(otu1)
potu2=convertOtuPhyloseq(otu2,samples[2])
head(potu2)
head(otu2)

## simulated data
set.seed(123)
comm = rsad(S=1000,frac=0.01,sad="lnorm",coef=list(meanlog=4,sdlog=2))
analyseOtu(data.frame(Count=comm))
diversity(comm)

## Read EBI tax table
setwd("~/src/git/djwhacks/r/metagenomics/zip/nfs/production/interpro/metagenomics/results/2017/05/DRP003216/version_3.0")
taxtab = read.delim("project-summary/taxonomy_abundances_v3.0.tsv",as.is=TRUE,header=TRUE,row.names=1)
head(taxtab)

## Read EBI phylum table
kptab = read.table("project-summary/phylum_taxonomy_abundances_v3.0.tsv",as.is=TRUE,header=TRUE)
rownames(kptab) = paste(kptab[,1],kptab[,2],sep=";")
kptab = kptab[,-(1:2)]
head(kptab)
dim(kptab)
## convert to phyloseq format
kpps = otu_table(kptab,taxa_are_rows=TRUE)
kpps
## create a dummy sample_data table
## TODO: Add in actual sample names...
sdtab = data.frame(Run=sample_names(kpps),
                   row.names=sample_names(kpps),
                   stringsAsFactors=FALSE)
sdps = sample_data(sdtab)
sdps
## combine otus and sample data
pseq = phyloseq(kpps,sdps)
pseq
## Try DESeq2
library(DESeq2)
packageVersion("DESeq2")
dedat = phyloseq_to_deseq2(pseq,~Run)
## FIT model...
defit = DESeq(dedat, test="Wald", fitType="parametric")
res = results(defit, cooksCutoff=FALSE)
res
res[res$padj < 0.05,]

results(defit, c("Run",sample_names(kpps)[3],sample_names(kpps)[5]),
        cooksCutoff=FALSE)$log2FoldChange

fchm=function(base,all=sample_names(kpps)){
    comp=all[all != base]
    mat = sapply(comp,
           function(x){
               results(defit,c("Run",base,x),cooksCutoff=FALSE)$log2FoldChange
           })
    rownames(mat)=rownames(kpps)
    mat=cbind(rep(1,dim(mat)[1]),mat)
    colnames(mat)[1]=base
    heatmap(mat,scale="none",main=paste("Fold-change compared to",base))
    invisible(mat)
}
fchm(sample_names(kpps)[3])
fchm(sample_names(kpps)[1])

## eof
