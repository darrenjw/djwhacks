package regression

import breeze.linalg.{ DenseMatrix, DenseVector }
import org.saddle._
import FrameUtils._

class ModelMatrix(df: Frame[Int, String, Double]) {

  val X = modelMatrix(df)
  val names = "(Intercept)" :: (df.colIx.toVec.toSeq.toList)

  def modelMatrix(df: Frame[Int, String, Double]): DenseMatrix[Double] = {
    val X = DenseMatrix.zeros[Double](df.numRows, df.numCols + 1)
    X(::, 0) := DenseVector.ones[Double](df.numRows)
    X(::, 1 to df.numCols) := frame2mat(df)
    X
  }

}

object ModelMatrix {

  def apply(df: Frame[Int, String, Double]): ModelMatrix = {
    new ModelMatrix(df)
  }

}
