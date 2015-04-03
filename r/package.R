# package.R

package=function(somepackage)
{
  cpackage <- as.character(substitute(somepackage))
  print(cpackage)
  if(!require(cpackage,character.only=TRUE)){
    install.packages(cpackage)
    library(cpackage,character.only=TRUE)
  }
}

# eof

