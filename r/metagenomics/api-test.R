## api-test.R
## test of the EBI metagemomics API (v0.2)

library(jsonlite)

baseURL = "https://www.ebi.ac.uk/metagenomics/api/v0.2"

## abstract out page combination
combinePages = function(url,ps=80) {
    if (grepl('?',url,fixed=TRUE))
        sep = "&"
    else
        sep = "?"
    firstURL =paste(url,paste0("page_size=",ps),sep=sep)
    first = fromJSON(firstURL)
    lastPage = first$meta$pagination$pages
    startURL = paste(url,paste0("page_size=",ps,"&page="),sep=sep)
    urls = paste0(startURL,1:lastPage)
    pages = lapply(urls,function(x) fromJSON(x)$data)
    rbind_pages(pages)
    }

## list of studies
studies = combinePages(paste(baseURL,"studies",sep="/"))
studies$id

## list of samples for a study
myStudy = "SRP047083"
samplesURL = paste(baseURL,paste0("samples?study_accession=",myStudy),sep="/")
samples = combinePages(samplesURL)
samples
samples$id

## list of runs for a sample
mySample = "SRS711891"
runsURL = paste(baseURL,paste0("runs?sample_accession=",mySample),sep="/")
runs = combinePages(runsURL)
runs$id
runs$links


## eof

