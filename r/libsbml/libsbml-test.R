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


## Now walk through building an SPN object
library(smfsb)
## Use model "m" already loaded...

## Species and initial amounts
P=list()
ns = m$getNumSpecies()
Mv = numeric(ns)
Mn = vector("character",ns)
for (i in 0:(ns-1)) {
    s = m$getSpecies(i)
    Mn[i+1] = s$getId()
    a = s$getInitialAmount()
    Mv[i+1] = a
}
names(Mv) = Mn
P$M = Mv
## Global parameters
nparm = m$getNumParameters()
GPv = vector("numeric",nparm)
GPn = vector("character",nparm)
if (nparm>0) {
    for (i in 0:(nparm-1)) {
        parm = m$getParameter(i)
        GPv[i+1] = parm$getValue()
        GPn[i+1] = parm$getId()
    }
    names(GPv) = GPn
}
## Reactions
nr = m$getNumReactions()
Pre = matrix(0,nrow=nr,ncol=ns)
colnames(Pre)=Mn
Post = matrix(0,nrow=nr,ncol=ns)
colnames(Post)=Mn
Rn = vector("character",ns)
KLv = vector("expression",nr)
LPl = vector("list",nr)
for (i in 0:(nr-1)) {
    r = m$getReaction(i)
    Rn[i+1] = r$getId()
    nPre = r$getNumReactants()
    if (nPre>0) {
        for (j in 0:(nPre-1)) {
            sr = r$getReactant(j)
            sto = sr$getStoichiometry()
            Pre[i+1, sr$getSpecies()] = sto
        }
    }
    nPost = r$getNumProducts()
    if (nPost>0) {
        for (j in 0:(nPost-1)) {
            sr = r$getProduct(j)
            sto = sr$getStoichiometry()
            Post[i+1,sr$getSpecies()] = sto
        }
    }
    kl = r$getKineticLaw()
    KLv[i+1] = parse(text=formulaToString(kl$getMath()))
    nparm = kl$getNumLocalParameters()
    if (nparm>0) {
        lpv = vector("numeric",nparm)
        lpn = vector("character",nparm)
        for (j in 0:(nparm-1)) {
            parm = kl$getLocalParameter(j)
            lpv[j+1] = parm$getValue()
            lpn[j+1] = parm$getId()
        }
        names(lpv) = lpn
        LPv[[i+1]] = lpv
    }
}
rownames(Pre)=Rn
rownames(Post)=Rn
P$Pre = Pre
P$Post = Post
P$h = function(x, t, gp = GPv) {
    with(as.list(c(x, gp)), {
        x = vector("numeric",nr)
        for (i in 1:nr) {
            x[i] = with(as.list(LPv[[i]]), eval(KLv[i]))
        }
        return(x)
        })
}



## Now see if we can use P to do a simulation...
cat("\nNow running a big Gillespie simulation - may take a while...\n")
plot(simTs(P$M,0,1000,0.1,StepGillespie(P)))




## eof

