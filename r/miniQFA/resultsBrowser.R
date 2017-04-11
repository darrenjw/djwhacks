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
pdf("mqfa-all.pdf",5,5)
plot(df$Predicted,df$Actual,pch=19,col=2,cex=0.2)
dev.off()
print(dim(df))

sub=df[df$Interaction != "No interaction",]
dim(sub)
sub=sub[sub$ORF1 != sub$ORF2,] # strip double deletions
print(dim(sub))
pdf("mqfa-int.pdf",5,5)
plot(sub$Predicted,sub$Actual,pch=19,col=2,cex=0.2)
dev.off()

library(igraph)
g=graph_from_data_frame(sub)
pdf("mqfa-dag.pdf",10,10)
plot(g)
dev.off()
plot(g, layout = layout.fruchterman.reingold)
pdf("mqfa-circ.pdf",8,8)
plot(g, layout = layout.circle)
dev.off()
m=get.adjacency(g)
image(as.matrix(m))

gu=as.undirected(g)
mu=as.matrix(get.adjacency(gu))
pdf("mqfa-adj.pdf",5,5)
image(mu,col=grey(1:0))
dev.off()
cl=cluster_fast_greedy(gu)
## cl=cluster_edge_betweenness(gu)
pdf("mqfa-adj-fg.pdf",5,5)
image(mu[order(cl$membership),order(cl$membership)],col=grey(1:0))
dev.off()
pdf("mqfa-ug-fg.pdf",10,10)
plot(cl,gu)
dev.off()

cl=cluster_spinglass(gu)
pdf("mqfa-adj-sg.pdf",5,5)
image(mu[order(cl$membership),order(cl$membership)],col=grey(1:0))
dev.off()
pdf("mqfa-ug-sg.pdf",10,10)
plot(cl,gu)
dev.off()




## eof


