## sbml2spn-test.R
## Test the sbml2spn parser on the auto-reg model

source("sbml2spn.R")

filename = "autoreg-3-1.xml"
N = sbml2spn(filename)

## Now see if we can use P to do a simulation...
cat("\nNow running a big Gillespie simulation - may take a while...\n")
plot(simTs(N$M,0,100,0.1,StepGillespie(N)))




## eof

