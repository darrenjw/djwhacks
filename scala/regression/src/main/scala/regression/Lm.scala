package statslang

import breeze.linalg.{ DenseMatrix, DenseVector }
import breeze.stats.regression._

class Lm(X: DenseMatrix[Double], y: DenseVector[Double]) {

  val ls = leastSquares(X, y)
  val beta = ls.coefficients

  override def toString: String = {
    beta.toString
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
 
 