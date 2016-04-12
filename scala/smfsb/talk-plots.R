# talk-plots.R
# Script to produce all of the plots for my INI talk...

library(smfsb)

# Functions to produce the actual plots

mcmcPlots=function(mat,root) {
    pdf(paste(root,"-tr.pdf",sep=""),10,8)
    mcmcSummary(mat)
    dev.off()
    pdf(paste(root,"-sc.pdf",sep=""),10,8)
    plot(mat,pch=19,cex=0.2,main=root)    
    dev.off()
}

abcPlots=function(mat,root) {
    mcmcSummary(mat,plot=FALSE)
    pdf(paste(root,"-md.pdf",sep=""),10,8)
    op=par(mfrow=c(2,2))
    names=colnames(mat)
    p=dim(mat)[2]
    for (i in 1:p) {
        hist(mat[,i],30,main=names[i],xlab="Value",freq=FALSE)
    }
    par(op)
    dev.off()
    pdf(paste(root,"-sc.pdf",sep=""),10,8)
    plot(mat,pch=19,cex=0.2,main=root)    
    dev.off()
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
    # colnames(nm)=c("k3","k4r","k5","k6")
    colnames(nm)=c("kTrans","kDiss","kRDeg","kPDeg")
    nm
}

myLog=function(mat) {
    names=colnames(mat)
    newNames=paste("log(",names,")",sep="")
    lm=log(mat)
    colnames(lm)=newNames
    lm
}

plotInf=function() {
    plotFile("AR-Pmmh10k-240-ct.csv")
    plotFile("AR-Abc1m.csv",type="abc")
    plotFile("AR-AbcSs1m.csv",type="abc")
    for (i in 1:10) {
        filename=sprintf("AR-AbcSmc10k-%03d.csv",i)
        plotFile(filename,type="abc")
    }
}

plotData=function() {
    perf=read.csv("AR-perfect.txt",header=FALSE)
    times=perf[,1]
    perf=ts(perf[,2:dim(perf)[2]],start=times[1],deltat=times[2]-times[1])
    pdf("AR-Data.pdf",10,8)
    plot(perf,plot.type="single",lty=c(2,2,2,1,1),ylab="Molecule count",lwd=2,main="True species counts at 50 time points and noisy data on two species")
    noise=read.csv("AR-noise10.txt",header=FALSE)
    points(times,noise[,5],pch=19)
    points(times,noise[,6],pch=17)
    dev.off()
}

# script to generate the plots

plotData()
plotInf()


# eof

