# quads.R

# plot quadrilaterals with vertices on a regular octogon

combs=combn(7,3)
n=dim(combs)[2]
combs=rbind(rep(0,n),combs,rep(0,n))

coord=function(n)
{
  c(cos(n*pi/4),sin(n*pi/4))
}

for (i in 1:n)
{
  plot(NULL,xlim=c(-1,1),ylim=c(-1,1),main=paste("Shape",i,"- Vertices:",combs[1,i],combs[2,i],combs[3,i],combs[4,i],sep=" "))
  #points(t(sapply(0:7,coord)),pch=19,col=3)
  text(t(sapply(0:7,coord)),labels=0:7)
  polygon(t(sapply(combs[,i],coord)),lwd=2,col="gray")
}

# eof


