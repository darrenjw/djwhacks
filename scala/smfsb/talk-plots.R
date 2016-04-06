# talk-plots.R
# Script to produce all of the plots for my INI talk...

library(smfsb)

# Functions to produce the actual plots

mcmcPlots=function(mat,root) {
    mcmcSummary(mat)
    plot(mat,pch=19,cex=0.2,main=root)    
}

abcPlots=function(mat,root) {
    mcmcSummary(mat,plot=FALSE)
    op=par(mfrow=c(2,2))
    names=colnames(mat)
    p=dim(mat)[2]
    for (i in 1:p) {
        hist(mat[,i],30,main=names[i],xlab="Value",freq=FALSE)
    }
    par(op)
    plot(mat,pch=19,cex=0.2,main=root)    
}

plotFile=function(filename,type="mcmc") {
    message("")
    message(paste(rep("=",65),collapse=""))    
    message(filename)
    root=strsplit(filename,".",fixed=TRUE)[[1]][1]
    message(root)
    message("")
    mat=myCols(read.csv(filename))
    if (type=="mcmc") {
        mcmcPlots(mat,root)
        mcmcPlots(myLog(mat),paste(root,"-log",sep=""))
    } else {
        abcPlots(mat,root)
        abcPlots(myLog(mat),paste(root,"-log",sep=""))
    }
}

myCols=function(mat) {
    nm=mat[,c("c3","c5","c6","c7")]
    colnames(nm)=c("k3","k4r","k5","k6")
    nm
}

myLog=function(mat) {
    names=colnames(mat)
    newNames=paste("log(",names,")",sep="")
    lm=log(mat)
    colnames(lm)=newNames
    lm
}

# Script to read the data and call the plotting functions

plotFile("AR-Pmmh10k-240-r1.csv")

plotFile("AR-Abc1m.csv",type="abc")
plotFile("AR-AbcSs1m.csv",type="abc")

for (i in 1:10) {
    filename=sprintf("AR-AbcSmc10k-%03d.csv",i)
    plotFile(filename,type="abc")
}

# eof

