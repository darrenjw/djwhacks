## function to simulate a simple game

p = function(m, n, a=0.3, b=0.4, N=100000) {
    c = 1-a-b
    succ = 0
    for (i in 1:N) {
        mm = 0
        nn = 0
        repeat {
            u = runif(1,0,1)
            if (u < a)
                mm = mm + 1
            else if (u < a+b)
                nn = nn + 1
            if (nn == n)
                break
        }
        if (mm == m)
            succ = succ + 1
    }
    succ/N
}


print( p(2,3) )

print( p(3, 3, a=0.5, b=0.4) )

## eof

