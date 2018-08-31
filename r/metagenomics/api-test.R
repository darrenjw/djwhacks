## api-test.R
## test of the EBI metagemomics API (v0.2)

library(jsonlite)


#### DEBUG
##fromJSON = function(url) {
##    message(url)
##    jsonlite::fromJSON(url)
##    }
#### END DEBUG


## R Client for new API

baseURL = "https://www.ebi.ac.uk/metagenomics/api/v1"

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

getStudies = function(...) {
    url1 = paste(baseURL,"studies",sep="/")
    dotList = list(...)
    if (length(dotList) == 0)
        url2 = url1
    else
        url2 = paste(url1,paste(...,sep="&"),sep="?")
    combinePages(url2)
}

getStudy = function(study){
    fromJSON(paste(baseURL,"studies",study,sep="/"))$data
}

getSamples = function(study) {
    samplesURL = paste(baseURL,paste0("samples?study_accession=",study),sep="/")
    combinePages(samplesURL)
}

getRunsBySample = function(sample) {
    runsURL = paste(baseURL,paste0("runs?sample_accession=",sample),sep="/")
    combinePages(runsURL)
}

getAnalysesByRun = function(run) {
    analysisURL = paste(baseURL,"runs",run,"analyses",sep="/")
    combinePages(analysisURL)
    }

getOtuByRun = function(run) {
    analyses = getAnalysesByRun(run)
    taxURL = as.character(analyses$relationships$taxonomy[[1]])
    tax = combinePages(taxURL)
    tax$attributes[,1:2]
    }


## Examples

studies = getStudies()
studies$id
getStudies("search=16S")$id
getStudies("centre_name=BioProject")$id
getStudies("centre_name=BioProject","search=16S")$id

myStudy = "SRP047083"
study = getStudy(myStudy)
study$id

samples = getSamples(myStudy)
samples
samples$id

mySample = "SRS711891"
runs = getRunsBySample(mySample)
runs$id
runs$links
runs

myRun = "SRR1589748"
analyses = getAnalysesByRun(myRun)
analyses
analyses$relationships$taxonomy
analyses$relationships$"taxonomy-ssu"
analyses$relationships$"taxonomy-lsu"

## taxsURL = as.character(analyses$relationships$"taxonomy-ssu"[[1]])
taxURL = as.character(analyses$relationships$taxonomy[[1]])
taxURL
tax = combinePages(taxURL)
tax$attributes[,1:2]

otu = getOtuByRun(myRun)
dim(otu)
str(otu)
barplot(otu$count)


## eof

