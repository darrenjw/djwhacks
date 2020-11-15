@model:3.1.1=Noise "Model of protein expression noise"
 s=item, t=second, v=litre, e=item
@compartments
 Cell
@parameters
 k1=1
 k2=0.5
 k3=10
 k4=0.1
@species
 Cell:Rna=0 s
 Cell:P=0 s
@reactions
@r=Transcription
 -> Rna
 k1
@r=RnaDegradation "RNA Degradation"
 Rna ->
 k2*Rna
@r=Translation
 Rna -> Rna + P
 k3*Rna
@r=ProteinDegradation "Protein degradation"
 P ->
 k4*P
