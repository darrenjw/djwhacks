package statslang

import breeze.linalg.{ DenseMatrix, DenseVector }
import com.github.fommil.netlib.BLAS.{ getInstance => blas }

class Lm(X: DenseMatrix[Double], y: DenseVector[Double]) {

  val qr = thinQR(X)
  val q = qr._1
  val r = qr._2
  val qty = q.t * y
  val fitted = q * qty
  val residuals = y - fitted
  val coefficients = backSolve(r, qty)

  override def toString: String = {
    coefficients.toString
  }

  def backSolve(A: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
    val ya = y.toArray
    blas.dtrsv("U", "N", "N", A.cols, A.toArray, A.rows, ya, 1)
    DenseVector[Double](ya)
  }

  def thinQR(A: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    import breeze.linalg
    val n = A.rows
    val p = A.cols
    val linalg.qr.QR(_Q, _R) = linalg.qr(X)
    (_Q(::, 0 until p), _R(0 until p, ::))
  }

}

object Lm {
  def apply(X: DenseMatrix[Double], y: DenseVector[Double]) = new Lm(X, y)
}
 
 