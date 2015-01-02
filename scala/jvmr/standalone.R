# standalone.R

system("sbt \"run output.csv 50000 1000\"")
out=read.csv("output.csv")
library(smfsb)
mcmcSummary(out,rows=2)

# eof





