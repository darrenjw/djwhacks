package regression

import breeze.linalg.{ DenseMatrix, DenseVector }
import org.saddle._
import org.saddle.io._

object FrameUtils {

  def frame2mat(df: Frame[Int, String, Double]): DenseMatrix[Double] = {
    // DenseMatrix(df.numRows,df.numCols,df.toMat.contents)
    // The above doesn't seem to work for some reason, so need to loop over columns instead...
    val X = DenseMatrix.zeros[Double](df.numRows, df.numCols)
    val M = df.toMat
    for (i <- 0 until df.numCols) {
      X(::, i) := DenseVector(M.takeCols(i).contents)
    }
    X
  }

  def getCol(colName: String, sdf: Frame[Int, String, String]): Frame[Int, String, Double] = {
    sdf.col(colName).mapValues(CsvParser.parseDouble)
  }

  def getColS(colName: String, sdf: Frame[Int, String, String]): Frame[Int, String, String] = {
    sdf.col(colName)
  }

}