# lattice-walk.R
# Demonstrate approximate isotropy of random walk on a lattice


plotWalk=function(s=10,p=10000)
{
 disp=matrix(sample(c("u","d","l","r"),s*p,replace=TRUE),nrow=p)
 up=(disp=="u")*1
 down=(disp=="d")*1
 left=(disp=="l")*1
 right=(disp=="r")*1
 xm=right-left
 ym=up-down
 x=apply(xm,1,sum)
 y=apply(ym,1,sum)
 plot(x,y,pch=19,main=paste("Path length",s))
}

op=par(mfrow=c(3,3))
lapply(c(5,6,9,10,20,50,100,200,500),plotWalk)
par(op)


# eof


