package regression

import breeze.linalg.{ DenseMatrix, DenseVector }
import org.saddle._
import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import FrameUtils._

class Lm(X: ModelMatrix, ys: Frame[Int, String, Double]) {

  println("h5")
  val y = frame2mat(ys)
  //println(y)
  val qr = thinQR(X.X)
  val q = qr._1
  //println(q)
  val r = qr._2
  //println(r)
  val qty = q.t * y
  //println(qty)
  val fitted = q * qty
  //println(fitted)
  val residuals = y - fitted
  val coefficients = backSolve(r, qty)

  override def toString: String = {
    coefficients.toString
  }

  def backSolve(A: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
    val ya = y.toArray
    blas.dtrsv("U", "N", "N", A.cols, A.toArray, A.rows, ya, 1)
    DenseVector(ya)
  }

  def backSolve(A: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ya = y.toArray
    blas.dtrsm("L", "U", "N","N", y.rows,y.cols,1.0, A.toArray, A.rows, ya, y.rows)
    DenseMatrix(ya)
  }

  // TODO: This is VERY inefficient for large n - need to replace with a proper thin QR by modifying qr() function definition
  def thinQR(A: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    import breeze.linalg
    val n = A.rows
    val p = A.cols
    val linalg.qr.QR(_Q, _R) = linalg.qr(A)
    (_Q(::, 0 until p), _R(0 until p, ::))
  }

}

object Lm {
  def apply(X: ModelMatrix, y: Frame[Int, String, Double]) = new Lm(X, y)
}
 
 