# makeData.R
# R code for generating data files from "smfsb" R package

# Need the following CRAN package:
# install.packages("smfsb")

require(smfsb)

data(LVdata)
write(LVpreyNoise10,"LVpreyNoise10.txt",ncolumns=1)

# eof


