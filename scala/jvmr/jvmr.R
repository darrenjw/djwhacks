# jvmr.R

sbtInit<-function()
{
  library(jvmr)
  system2("sbt","compile")
  cpstr=system2("sbt","printClasspath",stdout=TRUE)[3]
  cpsp=strsplit(cpstr,"!")[[1]]
  cp=cpsp[1:(length(cpsp)-1)]
  scalaInterpreter(cp)
}

sc=sbtInit()
sc['import gibbs.Gibbs._']
out=sc['genIters(State(0.0,0.0),10000,10).toArray.map{s=>Array(s.x,s.y)}']
library(smfsb)
mcmcSummary(out,rows=2)



# eof





