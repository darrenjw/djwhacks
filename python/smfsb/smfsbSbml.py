#!/usr/bin/env python3
# smfsbSbml.py
# Import SBML models into an smfsb Spn

import libsbml
import smfsb
import sys
import numpy as np

def file2Spn(filename, verb=False):
    d = libsbml.readSBML(filename)
    m = d.getModel()
    if (m == None):
        print("Can't parse SBML file: "+filename)
        sys.exit(1)
    return(model2Spn(m, verb))

def model2Spn(m, verb=False):
    # Species and initial amounts
    ns = m.getNumSpecies()
    if (verb):
        print(str(ns)+" species")
    ml = []
    nl = []
    for i in range(ns):
        s = m.getSpecies(i)
        nl += [s.getId()]
        ml += [s.getInitialAmount()]
    if (verb):
        print(nl)
        print(ml)
    # Compartments
    nc = m.getNumCompartments()
    cd = {}
    for i in range(nc):
        comp = m.getCompartment(i)
        cd[comp.getId()] = comp.getVolume()
    if (verb):
        print(cd)
    # Global parameters
    ngp = m.getNumParameters()
    gpd = {}
    for i in range(ngp):
        param = m.getParameter(i)
        gpd[param.getId()] = param.getValue()
    if (verb):
        print(gpd)
    # Reactions
    nr = m.getNumReactions()
    if (verb):
        print(str(nr)+" reactions")
    pre = np.zeros((nr, ns))
    post = np.zeros((nr, ns))
    rn = []
    kl = []
    lpl = []
    for i in range(nr):
        r = m.getReaction(i)
        rn += [r.getId()]
        nPre = r.getNumReactants()
        for j in range(nPre):
            sr = r.getReactant(j)
            sto = sr.getStoichiometry()
            pre[i, nl.index(sr.getSpecies())] = sto
        nPost = r.getNumProducts()
        for j in range(nPost):
            sr = r.getProduct(j)
            sto = sr.getStoichiometry()
            post[i, nl.index(sr.getSpecies())] = sto
        kli = r.getKineticLaw()
        kl += [libsbml.formulaToString(kli.getMath())]
        nlp = kli.getNumLocalParameters()
        lpd = {}
        for j in range(nlp):
            param = kli.getLocalParameter(j)
            lpd[param.getId()] = param.getValue()
        lpl += [lpd]
    if (verb):
        print(rn)
        print("Pre:")
        print(pre)
        print("Post:")
        print(post)
        print(kl)
        print(lpl)
    gpd.update(cd)
    def haz(x, t):
        h = np.zeros(nr)
        xd = dict(zip(nl, x))
        glob = gpd.copy()
        glob.update(xd)
        for i in range(nr):
            h[i] = eval(kl[i], glob, lpl[i])
        return(h)
    spn = smfsb.Spn(nl, rn, pre, post, haz, ml)
    spn.comp = cd
    spn.gp = gpd
    spn.kl = kl
    spn.lp = lpl
    return(spn)



# Test code

if (__name__ == '__main__'):
    spn = file2Spn("lambda.xml", True)
    print("\n\n\nModel created:\n\n")
    print(spn)
    print(spn.h(spn.m, 0))
    step = spn.stepGillespie()
    print(step(spn.m, 0, 20.0))
    out = smfsb.simTs(spn.m, 0, 100, 0.1, step)
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    for i in range(len(spn.m)):
        axis.plot(range(out.shape[0]), out[:,i])
    axis.legend(spn.n)
    plt.savefig("lambda.png")
    
# eof


