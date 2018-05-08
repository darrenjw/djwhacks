## libsbml-test.R

## See:
## http://sbml.org/Software/libSBML/libSBML_R_Example_Programs
## for some useful example programs



## load up the libSBML R bindings
library(libSBML)

## read in a test file
filename = "autoreg-3-1.xml"
d = readSBML(filename)

## process errors
errors = SBMLDocument_getNumErrors(d)
if (errors > 0) {
    SBMLDocument_printErrors(d)
    cat(paste("\n",errors,"ERRORS IN TOTAL"))
} else {
    cat("No errors")
    }

## extract model and print some basics
m = d$getModel()

level   = SBase_getLevel  (d)
version = SBase_getVersion(d)

cat("\n")
cat("File: ",filename," (Level ",level,", version ",version,")\n")

cat("         ")
cat("  model id: ", ifelse(Model_isSetId(m), Model_getId(m) ,"(empty)"),"\n")

cat( "functionDefinitions: ", Model_getNumFunctionDefinitions(m) ,"\n" )
cat( "    unitDefinitions: ", Model_getNumUnitDefinitions    (m) ,"\n" )
cat( "   compartmentTypes: ", Model_getNumCompartmentTypes   (m) ,"\n" )
cat( "        specieTypes: ", Model_getNumSpeciesTypes       (m) ,"\n" )
cat( "       compartments: ", Model_getNumCompartments       (m) ,"\n" )
cat( "            species: ", Model_getNumSpecies            (m) ,"\n" )
cat( "         parameters: ", Model_getNumParameters         (m) ,"\n" )
cat( " initialAssignments: ", Model_getNumInitialAssignments (m) ,"\n" )
cat( "              rules: ", Model_getNumRules              (m) ,"\n" )
cat( "        constraints: ", Model_getNumConstraints        (m) ,"\n" )
cat( "          reactions: ", Model_getNumReactions          (m) ,"\n" )
cat( "             events: ", Model_getNumEvents             (m) ,"\n" )
cat( "\n" )


## Now walk through model extracting key modelling info
cat(m$getId(),"\n")

## Species and initial amounts
ns = m$getNumSpecies()
cat(paste(ns,"species:\n"))
for (i in 0:(ns-1)) {
    s = m$getSpecies(i)
    cat(s$getId())
    a = s$getInitialAmount()
    cat(paste(" =",a))
    cat("\n")
}

## Global parameters
## TODO

## Reactions
nr = m$getNumReactions()
cat(paste(nr,"reactions:\n"))
for (i in 0:(nr-1)) {
    r = m$getReaction(i)
    cat(r$getId(),": ")
    nPre = r$getNumReactants()
    if (nPre>0) {
        for (j in 0:(nPre-1)) {
            if (j>0) cat(" + ")
            sr = r$getReactant(j)
            cat(sr$getStoichiometry(),"")
            cat(sr$getSpecies())
        }
    }
    cat(" -> ")
    nPost = r$getNumProducts()
    if (nPost>0) {
        for (j in 0:(nPost-1)) {
            if (j>0) cat(" + ")
            sr = r$getProduct(j)
            cat(sr$getStoichiometry(),"")
            cat(sr$getSpecies())
        }
    }
    cat(" ; ")
    kl = r$getKineticLaw()
    cat(formulaToString(kl$getMath()),"; ")
    nparm = kl$getNumLocalParameters()
    if (nparm>0) {
        for (j in 0:(nparm-1)) {
            parm = kl$getLocalParameter(j)
            cat(parm$getId(),"=",parm$getValue())
        }
    }
    cat("\n")
}


## N.B. eval(parse(text="x*2"))

## eof

