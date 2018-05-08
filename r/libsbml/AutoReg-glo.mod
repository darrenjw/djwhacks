@model:2.3.1=AutoRegulatoryNetwork
@compartments
 Cell=1
@species
 Cell:Gene=10
 Cell:P2Gene=0
 Cell:Rna=0
 Cell:P=0
 Cell:P2=0
@parameters
 k1=1
 k1r=10
 k2=0.01
 k3=10
 k4=1
 k4r=1
 k5=0.1
 k6=0.01
@reactions
@r=RepressionBinding
 Gene+P2 -> P2Gene
 k1*Gene*P2
@r=ReverseRepressionBinding
 P2Gene -> Gene+P2
 k1r*P2Gene
@r=Transcription
 Gene -> Gene+Rna
 k2*Gene 
@r=Translation
 Rna -> Rna+P
 k3*Rna 
@r=Dimerisation
 2P -> P2
 k4*0.5*P*(P-1) 
@r=Disassociation
 P2 -> 2P
 k4r*P2 
@r=RnaDegredation
 Rna ->
 k5*Rna 
@r=ProteinDegredation
 P ->
 k6*P 
