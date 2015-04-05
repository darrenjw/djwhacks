# makeData.R
# R code for generating data files from "smfsb" R package

package=function(somepackage)
{
  cpackage <- as.character(substitute(somepackage))
  if(!require(cpackage,character.only=TRUE)){
    install.packages(cpackage)
    library(cpackage,character.only=TRUE)
  }
}

package(smfsb)

data(LVdata)
write(LVpreyNoise10,"LVpreyNoise10.txt",ncolumns=1)

# eof


