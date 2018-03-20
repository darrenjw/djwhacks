## iteration.R
## Example of iterating functions


f1 = function(x) {
    2/x
}

f2 = function(x) {
    x*x + x - 2
}

f3 = function(x) {
    (x*x + 2) / (2*x)
}



iterate = function(f, text="", x=1) {
    plot(f,xlim=c(0,2.5),ylim=c(0,3),type="l",main=paste("Iterating x = f(x)",text),xlab=paste0("x (for x0 = ",x,")"),ylab="f(x)")
    abline(0,1)
    for (i in 1:10) {
        fx = f(x)
        lines(c(x,x,fx),c(x,fx,fx))
        x = fx
        }
}

pdf()
op=par(mfrow=c(2,2))
iterate(f1,"for f(x) = 2/x")
iterate(f2,"for f(x) = x^2+x+2",x=1.4)
iterate(f2,"for f(x) = x^2+x+2",x=1.42)
iterate(f3,"for f(x) = (x^2+2)/2x",x=0.5)
par(op)
dev.off()









## eof
