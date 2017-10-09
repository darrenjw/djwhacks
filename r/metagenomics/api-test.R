## api-test.R
## test of the EBI metagemomics API (v0.2)

library(jsonlite)

baseURL = "https://www.ebi.ac.uk/metagenomics/api/v0.2"

## list of studies
studiesURL = paste(baseURL,"studies?page_size=1000",sep="/")
print(studiesURL)
studies = fromJSON(studiesURL)
studies
studies$data$id
studies$data$links

## list of samples for a study
myStudy = "SRP047083"
samplesURL = paste(baseURL,paste("samples?page_size=1000&study_accession=",myStudy,sep=""),sep="/")
samples = fromJSON(samplesURL)
samples
samples$data$id
samples$data$links

## list of runs for a sample
mySample = "SRS711891"
runsURL = paste(baseURL,paste("runs?page_size=1000&sample_accession=",mySample,sep=""),sep="/")
runs = fromJSON(runsURL)
runs
runs$data$id
runs$data$links


## eof

