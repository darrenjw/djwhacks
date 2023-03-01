import cats.*
import cats.implicits.*

import breeze.linalg.*
import breeze.numerics.*

import munit.*

// Example unit tests
class TsqrUnitTests extends FunSuite:

  import Tsqr.*

  def matClose(a: DMD, b: DMD, tol: Double = 1.0e-5): Boolean =
    val diff = a - b
    (sum(diff *:* diff) < tol)


  val x = DenseMatrix.rand(100000, 50)

  val (q, r) = tsQR(x)

  test("q and r should combine to give x") {
    assert(matClose(q * r, x))
  }

  test("q should have orthonormal columns") {
    assert(matClose(q.t * q, DenseMatrix.eye[Double](q.cols)))
  }

  test("r should be upper triangular") {
    assert(matClose(r, upperTriangular(r)))
  }

  test("r should agree with breeze function up to signs") {
    assert(matClose(abs(r), abs(qr.justR(x))))
  }



// eof


