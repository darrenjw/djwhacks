# analysis.R

# add a "screen name" column to a dataset of tweets using a mapping file

package=function(somepackage)
{
  cpackage <- as.character(substitute(somepackage))
  if(!require(cpackage,character.only=TRUE)){
    install.packages(cpackage)
    library(cpackage,character.only=TRUE)
  }
}

package(mapdata)

#filename="../data/tweets.csv"
filename="../data/first1k.csv"
message("reading tweets from file")
tweets=read.csv(filename,stringsAsFactors=FALSE,colClasses="character")
message(paste(dim(tweets)[1],"tweets read from file"))
print(str(tweets))
message(paste(length(grep("@",tweets$value)),"tweets contain @"))
tweetUserIds=unique(sort(tweets$id))
message(paste(length(tweetUserIds),"unique users"))
tweetCount=table(tweets$id)
print(summary(as.vector(tweetCount)))
hist(tweetCount[tweetCount<30])
png("map.png",600,1000)
map('worldHires',
    c('UK', 'Ireland', 'Isle of Man','Isle of Wight'),
    xlim=c(-11,3), ylim=c(49,60.9))	
points(tweets$x,tweets$y,pch=19,col=rgb(0.5,0,0,0.05))
dev.off()

message("reading screenname mapping file")
mapping=read.csv("../data/mapping.csv",stringsAsFactors=FALSE,colClasses="character")
row.names(mapping)=mapping$id
#print(dim(mapping))

message("mapping screennames onto data frame")
# mapping["219674963",]$username
require(parallel)
options(mc.cores=detectCores())
screennames=unlist(mclapply(tweets$id,function(x) mapping[x,]$username))
#print(dim(tweets))
#print(dim(screennames))
df=cbind(tweets,screen_name=screennames)
write.csv(df,file="allTweets.csv")


# eof


