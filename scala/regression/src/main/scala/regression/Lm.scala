package regression

import breeze.linalg.{ DenseMatrix, DenseVector }
import org.saddle._
import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import FrameUtils._

class Lm(yf: Frame[Int, String, Double], X: ModelMatrix) {

  val names = X.names
  val y = frame2mat(yf)
  val qr = thinQR(X.X)
  val q = qr._1
  val r = qr._2
  val qty = q.t * y
  val fitted = q * qty
  val residuals = y - fitted
  val coefficients = backSolve(r, qty)
  val coeffFrame = mat2frame(coefficients, Index(X.names.toArray), yf.colIx)

  override def toString: String = coeffFrame.toString
  def summary: LmSummary = new LmSummary(this)

  def backSolve(A: DenseMatrix[Double], y: DenseMatrix[Double]): DenseMatrix[Double] = {
    val yc = y.copy
    blas.dtrsm("L", "U", "N", "N", y.rows, y.cols, 1.0, A.toArray, A.rows, yc.data, y.rows)
    yc
  }

  // TODO: This is VERY inefficient for large n - need to replace with a proper thin QR by modifying qr() function definition
  def thinQR(A: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    import breeze.linalg
    val n = A.rows
    val p = A.cols
    val linalg.qr.QR(_Q, _R) = linalg.qr(A)
    (_Q(::, 0 until p), _R(0 until p, ::))
  }

  // Not actually using this now, but will be useful for something... 
  def backSolve(A: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
    val yc = y.copy
    blas.dtrsv("U", "N", "N", A.cols, A.toArray, A.rows, yc.data, 1)
    yc
  }

}

object Lm {

  def apply(y: Frame[Int, String, Double], X: ModelMatrix): Lm = new Lm(y, X)

  def apply(y: Frame[Int, String, Double], XF: Frame[Int, String, Double]): Lm = Lm(y, ModelMatrix(XF))

  def apply(y: Frame[Int, String, Double], LXF: List[Frame[Int, String, Double]]): Lm = Lm(y, joinFrames(LXF))

}
 
 