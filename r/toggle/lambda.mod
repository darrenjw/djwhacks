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
  sigma = 0.5
  deg = 0.05 # guess - missing from the paper (gamma_m in paper)
@reactions
@r=CIdimer
  2CI -> CI2
  cidim*CI*(CI-1)/2 : cidim=1e6
@r=CIdiss
  CI2 -> 2CI
  cidis*CI2 : cidis=0.1
@r=CroDimer
  2Cro -> Cro2
  crodim*Cro*(Cro-1)/2 : crodim=1e6
@r=CroDiss
  Cro2 -> 2Cro
  crodiss*Cro2 : crodiss=0.1
@r=CIIdimer
  2CII -> CII2
  ciidim*CII*(CII-1)/2 : ciidim=1e6
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
  ccib*Prmr*CI2 : ccib=1e6
@r=CI2diss
  PrmrCI2 -> Prmr + CI2
  ccid*PrmrCI2 : ccid=0.1
@r=Cro2bind
  Prmr + Cro2 -> PrmrCro2
  ccrob*Prmr*Cro2 : ccrob=1e6
@r=Cro2diss
  PrmrCro2 -> Prmr + Cro2
  ccrod*PrmrCro2 : ccrod=0.1
@r=CII2bind
  Pre + CII2 -> PreCII2
  cciib*Pre*CII2 : cciib=1e6
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
		   

	
				   
