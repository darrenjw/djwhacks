#!/usr/bin/env python3
# lambda.py
# Simulate the lambda phage model, embedding in script as SBML-shorthand

# Serves as an illustration of the new "smfsb" python package (on PyPI)

import smfsb
import numpy as np
import matplotlib.pyplot as plt

sh = """
@model:3.1.1=lambda "lambda phage (Robb and Shahrezaei, 2014)"
  s=item, t=minute, v=litre, e=item
@compartments
  Cell=1e-15
@species
  Cell:CI=0 s
  Cell:CI2=0 s
  Cell:Cro=0 s
  Cell:Cro2=0 s
  Cell:CII=0 s
  Cell:CII2=0 s
  Cell:Prmr=1 s
  Cell:PrmrCI2=0 s
  Cell:PrmrCro2=0 s
  Cell:Pre=1 s
  Cell:PreCI2=0 s
  Cell:PreCII2=0 s
  Cell:mCI=0 s
  Cell:mCII=0 s
  Cell:mCro=0 s
@parameters
  nA = 6.023e23
  sigma = 0.5
  deg = 0.1
@reactions
@r=CIdimer
  2CI -> CI2
  cidim*CI*(CI-1)/(nA*Cell) : cidim=1e6
@r=CIdiss
  CI2 -> 2CI
  cidis*CI2 : cidis=0.1
@r=CroDimer
  2Cro -> Cro2
  crodim*Cro*(Cro-1)/(nA*Cell) : crodim=1e6
@r=CroDiss
  Cro2 -> 2Cro
  crodiss*Cro2 : crodiss=0.1
@r=CIIdimer
  2CII -> CII2
  ciidim*CII*(CII-1)/(nA*Cell) : ciidim=1e6
@r=CIIdiss
  CII2 -> 2CII
  ciidiss*CII2 : ciidiss=0.1
@r=CIdeg
  CI ->
  cideg*CI : cideg=0.04
@r=CroDeg
  Cro ->
  crodeg*Cro : crodeg=0.05
@r=CIIdeg
  CII ->
  ciideg*CII : ciideg=0.12
@r=CI2bind
  Prmr + CI2 -> PrmrCI2
  ccib*Prmr*CI2/(nA*Cell) : ccib=1e6
@r=CI2diss
  PrmrCI2 -> Prmr + CI2
  ccid*PrmrCI2 : ccid=0.1
@r=Cro2bind
  Prmr + Cro2 -> PrmrCro2
  ccrob*Prmr*Cro2/(nA*Cell) : ccrob=1e6
@r=Cro2diss
  PrmrCro2 -> Prmr + Cro2
  ccrod*PrmrCro2 : ccrod=0.1
@r=CII2bind
  Pre + CII2 -> PreCII2
  cciib*Pre*CII2/(nA*Cell) : cciib=1e6
@r=CII2diss
  PreCII2 -> Pre + CII2
  cciid*PreCII2 : cciid=0.1
@r=CIrtc
  PrmrCI2 -> PrmrCI2 + mCI
  cCItc*PrmrCI2 : cCItc=1.6
@r=CIrtc
  PreCI2 -> PreCI2 + mCI
  cCItc*PreCI2 : cCItc=1.2
@r=Crotc
  Prmr -> Prmr + mCro
  cCrotc*Prmr : cCrotc=0.8
@r=CIItc
  Prmr -> Prmr + mCII
  cciitc*Prmr : cciitc=0.8
@r=mCIdeg
  mCI ->
  deg*mCI
@r=mCrodeg
  mCro ->
  deg*mCro
@r=mCIIdeg
  mCII ->
  deg*mCII
@r=mCItl
  mCI -> mCI + CI
  sigma*mCI
@r=mCrotl
  mCro -> mCro + Cro
  sigma*mCro
@r=mCIItl
  mCII -> mCII + CII
  sigma*mCII
"""
#mod = smfsb.mod2Spn("lambda.mod")
mod = smfsb.sh2Spn(sh)
print(mod)
step = mod.stepGillespie()

# create and plot a single realisation of the process
out = smfsb.simTs(mod.m, 0, 100, 0.1, step)
fig, axis = plt.subplots()
for i in range(len(mod.m)):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(mod.n)
fig.savefig("lambda.pdf")

## Now look at the levels of CII at time 60
print("Sample at time 60. Please wait...")
out = smfsb.simSample(1000, mod.m, 0, 60, step)
cii = out[:,mod.n.index("CII")]
fig, axis = plt.subplots()
axis.hist(cii, 30)
axis.set_title("CII at time 60")
plt.savefig("cii.pdf")

## Now look at the _average_ levels of CII up to time 60
print("Looking now at average up to time 60. Please wait...")
n = 2500
v = np.zeros(n)
for i in range(n):
    out = smfsb.simTs(mod.m, 0, 60, 0.1, step)
    v[i] = np.mean(out[:,mod.n.index("CII")])
fig, axis = plt.subplots()
axis.hist(v, 50)
axis.set_title("CII averaged up to time 60")
plt.savefig("ciiA.pdf")


# eof

