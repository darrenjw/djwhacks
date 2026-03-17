## dlyap.R
## Solving the discrete lyapunov equation in R

## Solve AXA' - X + Q = 0  for X

## simulate a test A and Q
n = 10
A = matrix(rnorm(n*n), ncol=n)
Q = matrix(rnorm(n*n), ncol=n)

## function to test a solution
test_dlyap = function(A, Q, X, verb=TRUE, tol=1.0e-8) {
    Z = (A %*% X %*% t(A)) - X + Q
    n = sum(Z*Z)
    if (verb)
        print(n)
    n < tol
}

print("First try dlyap from netcontrol")
Xnc = netcontrol::dlyap(t(A), Q) # transpose A for some reason...
print(test_dlyap(A, Q, Xnc))
## only works if A is transposed...

print("Next try a direct kronecker solution")

dlyap_kron = function(A, Q)
    matrix(solve(diag(nrow(A)^2)-(A%x%A), as.vector(Q)), ncol=nrow(A))

Xk = dlyap_kron(A, Q)
print(test_dlyap(A, Q, Xk))

print("Next try conversion to a continuous lyapunov")

dlyap_con = function(A, Q) {
    n = nrow(A)
    B = solve(A + diag(n), A - diag(n))
    R = 0.5*(diag(n) - B) %*% Q %*% t(diag(n) - B)
    maotai::lyapunov(B, -R)
}

Xc = dlyap_con(A, Q)
print(test_dlyap(A, Q, Xc))

print("Next solve by eigen-decomposition")

dlyap_eig = function(A, Q) {
    n = nrow(A)
    eig = eigen(A)
    F = solve(eig$vectors, t(solve(eig$vectors, t(Q))))
    W = matrix(eig$values, nrow=n, ncol=n)
    Y = -F / (W*t(W) - 1)
    Re(eig$vectors %*% Y %*% t(eig$vectors))
}

Xe = dlyap_eig(A, Q)
print(test_dlyap(A, Q, Xe))


## eof

