## clyap.R
## Solving the continuous Lyapunov equation in R

## Solve AX + XA' + Q = 0  for X (symmetric Q)

## simulate a test A and Q
n = 10
A = matrix(rnorm(n*n), ncol=n)
Q = matrix(rnorm(n*n), ncol=n)
Q = Q %*% t(Q) # PSD Q

## function to test a solution
test_clyap = function(A, Q, X, verb=TRUE, tol=1.0e-8) {
    Z = (A %*% X) + (X %*% t(A)) + Q
    n = sum(Z*Z)
    if (verb)
        print(n)
    n < tol
}

print("Try maotai::lyapunov")
Xml = maotai::lyapunov(A, -Q) # -Q according to docs
print(test_clyap(A, Q, Xml))



## TODO: add a direct solver based on an asymmetric eigendecomposition of A





## eof

