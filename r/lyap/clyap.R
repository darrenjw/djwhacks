## clyap.R
## Solving the continuous Lyapunov equation in R

## Solve AX + XA' + Q = 0  for X

## simulate a test A and Q
n = 10
A = matrix(rnorm(n*n), ncol=n)
Q = matrix(rnorm(n*n), ncol=n)


## function to test a solution
test_clyap = function(A, Q, X, verb=TRUE, tol=1.0e-8) {
    Z = (A %*% X) + (X %*% t(A)) + Q
    n = sum(Z*Z)
    if (verb)
        print(n)
    n < tol
}

print("Try a direct kronecker solution")

clyap_k = function(A, Q) {
    n = nrow(A)
    mv = solve((diag(n) %x% A) + (A %x% diag(n)), -as.vector(Q))
    matrix(mv, ncol=n)
}

Xk = clyap_k(A, Q)
print(test_clyap(A, Q, Xk))


print("Try maotai::lyapunov")
Xml = maotai::lyapunov(A, -Q) # -Q according to docs
print(test_clyap(A, Q, Xml))

print("Try an eigen-decomposition solution")

clyap_e = function(A, Q) {
    n = nrow(A)
    eig = eigen(A)
    R = solve(eig$vectors, t(solve(eig$vectors, t(Q))))
    W = matrix(eig$values, nrow=n, ncol=n)
    Y = -R / (W + t(W))
    Re(eig$vectors %*% Y %*% t(eig$vectors))
}

Xe = clyap_e(A, Q)
print(test_clyap(A, Q, Xe))





## eof

