# standalone.R

library(smfsb)
system("sbt run")
out=read.csv("output.csv")
mcmcSummary(out,rows=2)

# eof





