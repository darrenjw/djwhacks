@model:3.1.0=AutoRegulatoryNetwork "Auto-regulatory network"
 s=item,t=second,v=litre,e=item
@compartments
 Cell
@species
 Cell:Gene=10.0s
 Cell:P2Gene=0.0s "P2.Gene"
 Cell:Rna=0.0s
 Cell:P=0.0s
 Cell:P2=0.0s
@reactions
@r=RepressionBinding "Repression binding"
 Gene+P2 -> P2Gene
 k1 * Gene * P2 : k1=1.0
@r=ReverseRepressionBinding "Reverse repression binding"
 P2Gene -> Gene+P2
 k1r * P2Gene : k1r=10.0
@r=Transcription
 Gene -> Gene+Rna
 k2 * Gene : k2=0.01
@r=Translation
 Rna -> Rna+P
 k3 * Rna : k3=10.0
@r=Dimerisation
 2P -> P2
 k4 * 0.5 * P * (P - 1) : k4=1.0
@r=Dissociation
 P2 -> 2P
 k4r * P2 : k4r=1.0
@r=RnaDegradation "RNA Degradation"
 Rna -> 
 k5 * Rna : k5=0.1
@r=ProteinDegradation "Protein degradation"
 P -> 
 k6 * P : k6=0.01
