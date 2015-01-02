# breeze.R
# create a scala interpreter with breeze loaded into the classpath from scratch

source("breezeInit.R")

sc=breezeInit()
sc['import breeze.stats.distributions._']
sc['Poisson(10).sample(20).toArray']



# eof





