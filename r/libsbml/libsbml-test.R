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
m = SBMLDocument_getModel(d)

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
cat(Model_getId(m))
cat("\n")

## Species and initial amounts
ns = Model_getNumSpecies(m)
cat(paste(ns,"species:\n"))
for (i in 0:(ns-1)) {
    s = Model_getSpecies(m,i)
    cat(Species_getId(s))
    a = Species_getInitialAmount(s)
    cat(paste(" =",a))
    cat("\n")
}

## Global parameters
## TODO

## Reactions
nr = Model_getNumReactions(m)
cat(paste(nr,"reactions:\n"))
for (i in 0:(nr-1)) {
    r = Model_getReaction(m,i)
    cat(Reaction_getId(r),": ")
    nPre = Reaction_getNumReactants(r)
    if (nPre>0) {
        for (j in 0:(nPre-1)) {
            if (j>0) cat(" + ")
            sr = Reaction_getReactant(r,j)
            cat(SpeciesReference_getStoichiometry(sr))
            ##cat(SpeciesReference_getSpecies(sr))
        }
    }
    cat(" -> ")
    nPost = Reaction_getNumProducts(r)
    if (nPost>0) {
        for (j in 0:(nPost-1)) {
            if (j>0) cat(" + ")
            sr = Reaction_getProduct(r,j)
            cat(SpeciesReference_getStoichiometry(sr))
            ##cat(SpeciesReference_getSpecies(sr))
        }
    }
    cat(" ; ")
    kl = Reaction_getKineticLaw(r)
    cat(formulaToString(KineticLaw_getMath(kl)),"; ")
    nparm = KineticLaw_getNumLocalParameters(kl)
    if (nparm>0) {
        for (j in 0:(nparm-1)) {
            parm = KineticLaw_getLocalParameter(kl,j)
            cat(Parameter_getId(parm),"=",Parameter_getValue(parm))
        }
    }
    cat("\n")
}


## N.B. eval(parse(text="x*2"))

## eof

