## csv2md.R
## Convert a CSV file resulting from a web form into a readable markdown document


file = "apps.csv"

csv = read.csv(file, stringsAsFactor=FALSE)
fields = names(csv)

for (i in 1:dim(csv)[1]) {
    for (j in 1:length(fields)) {
        if (j != 1) cat("#")
        cat("## ")
        cat(fields[j])
        cat("\n\n")
        cat(csv[i,j])
        cat("\n\n")
        }
    cat("\n\n")
    }

## eof

