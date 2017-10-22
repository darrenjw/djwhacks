## rjsonapi-test.R
## Test of the api using rjsonapi instead of plain jsonlite...
## rjsonapi uses jsonlite under-the-hood, so not that much benefit...

library(rjsonapi)

baseURL = "https://www.ebi.ac.uk"
APIVersion = "metagenomics/api/v0.2"
conn = jsonapi_connect(baseURL,APIVersion)

#### DEBUG
## conn = jsonapi_connect(baseURL,APIVersion,verbose=TRUE)
#### END DEBUG

combinePages = function(route,ps=80) {
    if (grepl('?',route,fixed=TRUE))
        sep = "&"
    else
        sep = "?"
    firstRoute =paste(route,paste0("page_size=",ps),sep=sep)
    first = conn$route(firstRoute)
    lastPage = first$meta$pagination$pages
    if (lastPage == 1)
        first$data
    else {
        startRoute = paste(route,paste0("page_size=",ps,"&page="),sep=sep)
        urls = paste0(startRoute,2:lastPage)
        pages = lapply(urls,function(x) conn$route(x)$data)
        jsonlite::rbind_pages(c(list(first$data),pages))
    }
}

getStudies = function(...) {
    route = "studies"
    dotList = list(...)
    if (length(dotList) == 0)
        url = route
    else
        url = paste(route,paste(...,sep="&"),sep="?")
    combinePages(url)
}

getStudy = function(study){
    conn$route(paste("studies",study,sep="/"))$data
}

getSamples = function(study) {
    samplesRoute = paste0("samples?study_accession=",study)
    combinePages(samplesRoute)
}

getRunsBySample = function(sample) {
    runsRoute = paste0("runs?sample_accession=",sample)
    combinePages(runsRoute)
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



## Debug stuff

##conn
##conn$opts
##conn$version
##conn$url
##conn$base_url()
##conn$status()
##conn$routes()
##studies = conn$route("studies")
##conn$route("studies?page_size=10&page=2")
##conn$route("samples?study_accession=ERP024243")$data


## eof

