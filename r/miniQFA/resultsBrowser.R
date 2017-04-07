## resultsBrowser
## Query Keith's results


library(jsonlite)

## Web browser:
## https://www.students.ncl.ac.uk/keith.newman/phd/miniqfa/results/20170117/

## Plot:
## https://www.students.ncl.ac.uk/keith.newman/phd/miniqfa/results/20170117/mq20170117_r_cdc13d127_gisPlot.pdf

## Table
url="https://www.students.ncl.ac.uk/keith.newman/phd/miniqfa/results/20170117/mq20170117_r_cdc13d127_gisTable.json"

rt = fromJSON(url)
df=data.frame(ORF1=rt[,1],ORF2=rt[,2],Pair=rt[,3],logdg=as.numeric(rt[,4]),gamma=as.numeric(rt[,5]),Predicted=as.numeric(rt[,6]),Actual=as.numeric(rt[,7]),Interaction=rt[,8],stringsAsFactors=FALSE)
plot(df$Predicted,df$Actual)

sub=df[df$Interaction != "No interaction",]
dim(sub)
sub=sub[sub$ORF1 != sub$ORF2,] # strip double deletions
dim(sub)
plot(sub$Predicted,sub$Actual)

library(igraph)
g=graph_from_data_frame(sub)
plot(g)
plot(g, layout = layout.fruchterman.reingold)
plot(g, layout = layout.circle)
m=get.adjacency(g)
image(as.matrix(m))
cluster_fast_greedy(g)





## eof


