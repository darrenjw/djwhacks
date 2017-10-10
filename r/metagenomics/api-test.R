## api-test.R
## test of the EBI metagemomics API (v0.2)

library(jsonlite)
baseURL = "https://www.ebi.ac.uk/metagenomics/api/v0.2"

## DEBUG
fromJSON = function(url) {
    message(url)
    jsonlite::fromJSON(url)
    }
## END DEBUG

## abstract out page combination
combinePages = function(url,ps=80) {
    if (grepl('?',url,fixed=TRUE))
        sep = "&"
    else
        sep = "?"
    firstURL =paste(url,paste0("page_size=",ps),sep=sep)
    first = fromJSON(firstURL)
    lastPage = first$meta$pagination$pages
    if (lastPage == 1)
        first$data
    else {
        startURL = paste(url,paste0("page_size=",ps,"&page="),sep=sep)
        urls = paste0(startURL,2:lastPage)
        pages = lapply(urls,function(x) fromJSON(x)$data)
        rbind_pages(c(list(first$data),pages))
    }
}

## list of studies
getStudies = function(...) {
    url1 = paste(baseURL,"studies",sep="/")
    dotList = list(...)
    if (length(dotList) == 0)
        url2 = url1
    else
        url2 = paste(url1,paste(...,sep="&"),sep="?")
    combinePages(url2)
}
## examples
studies = getStudies()
studies$id
getStudies("search=16S")$id
getStudies("centre_name=BioProject")$id
getStudies("centre_name=BioProject","search=16S")$id

## study
getStudy = function(study){
    
    }

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

