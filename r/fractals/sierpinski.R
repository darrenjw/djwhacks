## sierpinski.R

plot(NULL,xlim=c(0,1),ylim=c(0,1),xlab="",ylab="")

sierp = function(x, y, n) {
    if (n == 0)
        polygon(x, y, col=2)
    else {
        sierp(c(x[1],(x[1]+x[3])/2,(x[1]+x[2])/2),
              c(y[1],(y[1]+y[3])/2,(y[1]+y[2])/2), n-1)
        sierp(c(x[2],(x[1]+x[2])/2,(x[2]+x[3])/2),
              c(y[2],(y[1]+y[2])/2,(y[2]+y[3])/2), n-1)
        sierp(c(x[3],(x[1]+x[3])/2,(x[2]+x[3])/2),
              c(y[3],(y[1]+y[3])/2,(y[2]+y[3])/2), n-1)
    }       
}

sierp(c(0,1,0.5), c(0,0,1), 6)

for (i in 1:6) {
    plot(NULL,xlim=c(0,1),ylim=c(0,1),xlab="",ylab="")
    sierp(c(0,1,0.5), c(0,0,1), i)
    Sys.sleep(1)
}

## eof

