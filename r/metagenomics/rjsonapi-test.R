## rjsonapi-test.R
## test of the api using rjsonapi instead of jsonlite...

library(rjsonapi)

baseURL = "https://www.ebi.ac.uk"
APIVersion = "metagenomics/api/v0.2"
conn = jsonapi_connect(baseURL,APIVersion)
conn
conn$version
conn$url
conn$base_url()
conn$status()
conn$routes()







## eof

