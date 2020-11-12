@model:3.1.1=Dogma "Model of the central dogma"
 s=item, t=second, v=litre, e=item
@compartments
 Cell
@species
 Cell:Rna=0 s
 Cell:P=0 s
@reactions
@r=Transcription
 -> Rna
 k1 : k1=10
@r=RnaDegradation "RNA Degradation"
 Rna ->
 k3*Rna : k3=5
@r=Translation
 Rna -> Rna + P
 k2*Rna : k2=10
@r=ProteinDegradation "Protein degradation"
 P ->
 k4*P : k4=0.1
