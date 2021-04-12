# Smile basics

## Special functions

```scala mdoc
import smile.math.MathEx._
logistic(2.0)

import smile.math.special._
Erf.erf(1.0)

```

## Linear algebra


```scala mdoc
// c() is in MathEx
val x = c(1.0, 2.0, 3.0)
val y = c(3.0, 2.0, 2.0)

import smile.math.{matrix => pmatrix, _}
val z = x + y
z
cov(x, y)

import smile.math.matrix._
val m = matrix(c(1.0, 2.0), c(3.0, 4.0))
m
m.diag
m.trace

eye(3)
zeros(2, 3)
ones(3)
ones(3, 4) // doesn't work! BUG! Currently returns matrix of zeros.

// back and forth between matrices and nested arrays
val arr = Array( Array(1.0, 2.0), Array(3.0, 4.0))
matrix(arr)
~arr
matrix(arr).toArray

// matrix vector mult
m * c(1.0,2.0)
// matrix matrix
m %*% m // need this op or get hadamard

// solve
m \ c(1.0,2.0)

// svd
m.svd // or:
val s = svd(m)
s.U
s.V
s.diag
s.s
s.U %*% s.diag %*% s.V.t
// a multiplication chain will be analysed and optimised

// random normal matrix
val nmat = randn(5,5)
nmat.normFro // Frobenious norm?

// cholesky
val m5 = nmat %*% nmat.t
val m5s = new SymmMatrix(smile.math.blas.UPLO.LOWER,m5.toArray)
// m5s.isSymmetric // Not defined!
val lu = m5s.cholesky().lu // both triangles...

val m5m = nmat.aat()
m5m.isSymmetric
m5m.cholesky().lu // just want lower...


// eigen
val eig = m5.eigen
eig.Vr // right eigenvectors
eig.Vl // NULL - left eigenvectors
eig.diag // diag matrix of eigenvalues
eig.wr // real parts of eigenvalues
eig.wi // imaginary parts of eigenvalues (zero)
eig.Vr %*% eig.diag %*% eig.Vr.t

// qr
val mTall = randn(10,3)
val qrdecomp = mTall.qr
qrdecomp.Q
qrdecomp.R
qrdecomp.Q %*% qrdecomp.R  // No!
qrdecomp.Q.t %*% qrdecomp.Q // No!

```

## Statistics

```scala mdoc:reset
val v = (1 to 10).toArray

import smile.math.MathEx._
import smile.stat._
mean(v)
sd(v)
max(v)
whichMax(v)

import smile.stat.distribution._
val expd = new ExponentialDistribution(2.0)
expd.mean
expd.variance
expd.rand()
expd.rand(10)
expd.p(1.0)
expd.cdf(1.0)
expd.quantile(0.5)
expd.logLikelihood(Array(1.0))
expd.logLikelihood(Array(1.0, 2.0, 1.5))
```
