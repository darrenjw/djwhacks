## tree.R
## pythagorean tree

plot(NULL,xlim=c(0,1),ylim=c(0,1),xlab="",ylab="")

tree = function(x, y, n) {
    x3 = x[1] - (y[2]-y[1])
    y3 = y[1] + (x[2]-x[1])
    x4 = x3 + (x[2]-x[1])
    y4 = y3 + (y[2]-y[1])
    xp = c(x4,x3)
    yp = c(y4,y3)
    polygon(c(x,xp), c(y,yp), col=2)
    if (n > 0) {
        x5 = 0.5*(x3+x4) - 0.5*(y4-y3)
        y5 = 0.5*(y3+y4) + 0.5*(x4-x3)
        tree(c(x3,x5), c(y3,y5), n-1)
        tree(c(x5,x4), c(y5,y4), n-1)
    }       
}

tree(c(0.42,0.58), c(0,0), 10)

## eof

