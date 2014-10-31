package statslang

import breeze.linalg.{ DenseMatrix, DenseVector }
import breeze.stats.regression._

class Lm(X: DenseMatrix[Double], y: DenseVector[Double]) {

  val ls = leastSquares(X, y)
  val beta = ls.coefficients

  override def toString: String = {
    beta.toString
  }

}

 object Lm {
    def apply(X: DenseMatrix[Double], y: DenseVector[Double]) = new Lm(X, y)
  }
 
 