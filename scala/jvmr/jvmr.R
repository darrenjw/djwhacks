# jvmr.R

sbtInit<-function()
{
  library(jvmr)
  system2("sbt","compile")
  cpstr=system2("sbt","printClasspath",stdout=TRUE)
  cpst=cpstr[length(cpstr)]
  cpsp=strsplit(cpst,"!")[[1]]
  cp=cpsp[1:(length(cpsp)-1)]
  scalaInterpreter(cp,use.jvmr.class.path=FALSE)
}

sc=sbtInit()
sc['import gibbs.Gibbs._']
out=sc['genIters(State(0.0,0.0),50000,1000).toArray.map{s=>Array(s.x,s.y)}']
library(smfsb)
mcmcSummary(out,rows=2)



# eof





