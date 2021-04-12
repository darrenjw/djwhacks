# Failing test of QR

```scala mdoc

import smile.math.{matrix => pmatrix, _}
import smile.math.MathEx._
import smile.math.special._
import smile.math.matrix._

val mat = matrix(c(3.0,3.5),c(2.0,2.0),c(0.0,1.0))
val qrd = mat.qr()
//assert((qrd.Q.t %*% qrd.Q - eye(2)).normFro() < 1e-5)
//assert((qrd.Q %*% qrd.R - mat).normFro() < 1e-5)

```
