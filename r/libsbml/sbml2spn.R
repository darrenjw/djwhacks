## sbml2spn.R
## Parse an SBML file into a smfsb SPN object

library(libSBML)
library(smfsb)

sbml2spn = function(filename) {
    d = readSBML(filename)
    m = d$getModel()
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
            LPl[[i+1]] = lpv
        }
    }
    rownames(Pre)=Rn
    rownames(Post)=Rn
    P$Pre = Pre
    P$Post = Post
    P$GP = GPv
    P$KL = KLv
    P$LP = LPl
    P$h = function(x, t, gp = GPv, lp = LPl) {
        with(as.list(c(x, gp)), {
            x = vector("numeric",nr)
            for (i in 1:nr) {
                x[i] = with(as.list(lp[[i]]), eval(KLv[i]))
            }
            return(x)
        })
    }
    P
}



## eof

