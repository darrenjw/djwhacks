# lhd.R

# latin hypercube designs

# If SLHD not installed, install by executing:
# install.packages("SLHD")

library(SLHD)
plot(maximinSLHD(1,50,2)$D/50,pch=19,xlab="x1",ylab="x2",main="LHD for 2 factors with 50 design points")


# eof


