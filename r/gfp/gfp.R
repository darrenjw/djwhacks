# gfp.R
# very simple GFP fluorescence model

package=function(somepackage)
{
  cpackage <- as.character(substitute(somepackage))
  if(!require(cpackage,character.only=TRUE)){
    install.packages(cpackage)
    library(cpackage,character.only=TRUE)
  }
}
package(smfsb)
package(deSolve)

rhs <- function(s,t,parms)
{
  with(as.list(c(s,parms)),{
    c( r*x*(k-x) , a*x )
  })
}

stepGfp = StepODE(rhs)

out = simTs(c(x=1,y=0.015),0,50,0.1,stepGfp,parms=c(r=0.002,k=100,a=0.002))
lx = log(out[,"x"])
g = 50*out[,"y"]/out[,"x"]
out = cbind(out,lod=lx,g=g)
out = out[,3:4]
plot(out,plot.type="single",lwd=3,col=c("grey","red"))
legend(0,4,c("log(OD)","GFP"),col=c("grey","red"),lwd=c(3,3))


# eof


