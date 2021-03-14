## sphere.R

rsphere = function(r) {
    repeat {
        v = runif(3,-r,r)
        r2 = sum(v*v)
        if (r2 < r*r)
            return(v)
    }
}

mySphere = function()
    rsphere(0.5)  + c(0.5,0,0)

N = 1e6
sm = sapply(1:N, function(x) mySphere())
rv = apply(sm,2,function(x) sqrt(sum(x*x)))
m = mean(rv)
V = (4/3)*pi*0.5^3
int = m * V
int


## eof
