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
  kdim = 1e6
  kdis = 0.1
  cideg=0.04
  crodeg=0.05
  ciideg=0.12
  kon = 1e6
  koff = 0.1
  beta=1.6
  delta=1.2
  alpha=0.8
  deg = 0.1
  sigma = 0.5
@reactions
@r=CIdimer
  2CI -> CI2
  kdim*CI*(CI-1)/(nA*Cell)
@r=CIdiss
  CI2 -> 2CI
  kdis*CI2
@r=CroDimer
  2Cro -> Cro2
  kdim*Cro*(Cro-1)/(nA*Cell)
@r=CroDiss
  Cro2 -> 2Cro
  kdis*Cro2
@r=CIIdimer
  2CII -> CII2
  kdim*CII*(CII-1)/(nA*Cell)
@r=CIIdiss
  CII2 -> 2CII
  kdis*CII2
@r=CIdeg
  CI ->
  cideg*CI
@r=CroDeg
  Cro ->
  crodeg*Cro
@r=CIIdeg
  CII ->
  ciideg*CII
@r=CI2bind
  Prmr + CI2 -> PrmrCI2
  kon*Prmr*CI2/(nA*Cell)
@r=CI2diss
  PrmrCI2 -> Prmr + CI2
  koff*PrmrCI2
@r=Cro2bind
  Prmr + Cro2 -> PrmrCro2
  kon*Prmr*Cro2/(nA*Cell)
@r=Cro2diss
  PrmrCro2 -> Prmr + Cro2
  koff*PrmrCro2
@r=CII2bind
  Pre + CII2 -> PreCII2
  kon*Pre*CII2/(nA*Cell)
@r=CII2diss
  PreCII2 -> Pre + CII2
  koff*PreCII2
@r=CIrtc
  PrmrCI2 -> PrmrCI2 + mCI
  beta*PrmrCI2
@r=CIIrtc
  PreCII2 -> PreCII2 + mCI
  delta*PreCII2
@r=Crotc
  Prmr -> Prmr + mCro
  alpha*Prmr
@r=CIItc
  Prmr -> Prmr + mCII
  alpha*Prmr
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
				 