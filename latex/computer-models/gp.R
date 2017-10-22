# gp.R
# Script to demo simple GP emulation for 1 input and 1 output

# Setup
Grid=seq(0,1,0.01)

# GP covariance kernel
K=function(d,scale=40,lengthScale=0.4)
{
  (scale^2)*exp(-(d/lengthScale)^2)
}

# Fit a mean zero GP
GPAdjust=function(Grid,ObsLoc,Obs) 
{
  lG=length(Grid)
  lO=length(ObsLoc)
  All=c(Grid,ObsLoc)
  Dist=as.matrix(dist(All))
  Var=K(Dist)
  VG=Var[1:lG,1:lG]
  VO=Var[(lG+1):(lG+lO),(lG+1):(lG+lO)]
  CovGO=Var[1:lG,(lG+1):(lG+lO)]
  CovOG=t(CovGO)
  ViC=solve(VO,CovOG)
  AdjEx=t(ViC)%*%Obs
  AdjVar=VG-CovGO%*%ViC
  AdjVar=AdjVar+diag(rep((1e-6)*max(AdjVar),lG)) # numerical stability fudge
  list(Ex=as.vector(AdjEx),Var=AdjVar)
}

# Fit a GP to the residuals from a linear regression fit
GPAdjustLR=function(Grid,ObsLoc,Obs) 
{
  Mod=lm(Obs~ObsLoc)
  Pred=predict(Mod,newdata=data.frame(ObsLoc=Grid))
  Raw=GPAdjust(Grid,ObsLoc,Mod$residuals)
  list(Ex=Raw$Ex+Pred,Var=Raw$Var)
}

# Produce a plot illustrating the GP fit
plotAdjust=function(Grid,ObsLoc,Obs,...)
{
  plot(ObsLoc,Obs,xlim=range(Grid),pch=19,xlab="x",ylab="y",...)
  Adj=GPAdjustLR(Grid,ObsLoc,Obs)
  Var=diag(Adj$Var)
  Upper=Adj$Ex+2*sqrt(Var)
  Lower=Adj$Ex-2*sqrt(Var)
  polygon(c(Grid,rev(Grid),Grid[1]),c(Upper,rev(Lower),Upper[1]),col="gray",border=NA)
  lines(Grid,Adj$Ex,lwd=2)
  points(ObsLoc,Obs,pch=19)
  Adj
}

# Generate a sample from an adjusted object
sampleGP=function(Adj)
{
  L=t(chol(Adj$Var))
  Adj$Ex + L%*%rnorm(length(Adj$Ex))
}

# Now actually do some stuff...
ObsLoc=c(0.3,0.6,0.4,0.9,0.1,0.8)
Obs=c(10,11,8,12,12,13)

# Show data progressively
for (i in 1:length(Obs)) {
  pdf(paste("sim",i,".pdf",sep=""))
  plot(ObsLoc[1:i],Obs[1:i],xlim=range(Grid),pch=19,ylim=c(0,20),xlab="x",ylab="y")
  dev.off()
  }

# Show a progressive GP fit
for (i in 1:length(Obs)) {
  pdf(paste("emu",i,".pdf",sep=""))
  plotAdjust(Grid,ObsLoc[1:i],Obs[1:i],ylim=c(0,20))
  dev.off()
  }

# Now show some samples from a GP
for (i in 10^(0:2)) {
  pdf(paste("sample",i,".pdf",sep=""))
  plot(ObsLoc,Obs,xlim=range(Grid),pch=19,ylim=c(0,20),xlab="x",ylab="y")
  Adj=GPAdjustLR(Grid,ObsLoc,Obs)
  for (j in 1:i)
    lines(Grid,sampleGP(Adj),col="gray")
  points(ObsLoc,Obs,pch=19)
  dev.off()
  }

# History matching using implausibility
Measurement=10
MeasurementError=0.1
for (i in 1:length(Obs)) {
  pdf(paste("imp",i,".pdf",sep=""))
  op=par(mfrow=c(2,1))
  Adj=plotAdjust(Grid,ObsLoc[1:i],Obs[1:i],ylim=c(0,20))
  abline(Measurement,0,col=2,lwd=2)
  Imp=sqrt((Adj$Ex-Measurement)^2/(diag(Adj$Var)+MeasurementError^2))
  plot(Grid,Imp,ylim=c(0,10),type="l",lwd=2,xlab="x",ylab="Implausibility")
  abline(3,0,col=3)
  par(op)
  dev.off()
  }

# Bias example
ObsLoc=seq(0,1,0.13)
Obs=exp(-ObsLoc)
pdf("disc.pdf")
plotAdjust(Grid,ObsLoc,Obs,ylim=c(0,1))
abline(0.6,0,col=2,lwd=2)
points(rep(c(0.2,0.5,0.8),each=3),c(0.85,0.87,0.9,0.62,0.66,0.68,0.48,0.51,0.55),pch=4)
dev.off()

# eof


