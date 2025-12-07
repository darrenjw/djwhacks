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

## start with a simple kronecker implementation
clyap_k = function(A, Q) {
    n = nrow(A)
    mv = solve((diag(n) %x% A) + (A %x% diag(n)), -as.vector(Q))
    matrix(mv, ncol=n)
}

## check that it works
print("Testing clyap_k")
Xk = clyap_k(A, Q)
print(test_clyap(A, Q, Xk))

## check that the maotai implementation works
print("Try maotai::lyapunov")
Xml = maotai::lyapunov(A, -Q) # -Q according to docs
print(test_clyap(A, Q, Xml))



## TODO: add a direct solver based on an asymmetric eigendecomposition of A





## eof

