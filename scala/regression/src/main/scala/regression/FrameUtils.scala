package regression

import breeze.linalg.{ DenseMatrix, DenseVector }
import org.saddle._
import org.saddle.io._

object FrameUtils {

  def frame2mat[T](df: Frame[T, String, Double]): DenseMatrix[Double] = {
    // DenseMatrix(df.numRows,df.numCols,df.toMat.contents)
    // TODO: The above doesn't seem to work for some reason, so loop over columns instead as a temp hack
    val X = DenseMatrix.zeros[Double](df.numRows, df.numCols)
    val M = df.toMat
    for (i <- 0 until df.numCols) {
      X(::, i) := DenseVector(M.takeCols(i).contents)
    }
    X
  }

  def mat2frame(M: DenseMatrix[Double], rowIx: Index[String], colIx: Index[String]): Frame[String, String, Double] = {
    val SM = Mat(M.rows, M.cols, M.t.toArray)
    Frame(SM, rowIx, colIx)
  }

  def joinFrames[T](frames: List[Frame[T, String, Double]]): Frame[T, String, Double] = {
    // TODO: Use pattern matching!
    if (frames.length == 1) frames.head else
      frames.head.joinPreserveColIx(joinFrames(frames.tail))
  }

  def getCol[T](colName: String, sdf: Frame[T, String, String]): Frame[T, String, Double] = {
    sdf.col(colName).mapValues(CsvParser.parseDouble)
  }

  def getColS[T](colName: String, sdf: Frame[T, String, String]): Frame[T, String, String] = {
    sdf.col(colName)
  }

}