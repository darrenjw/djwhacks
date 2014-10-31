
package statslang

object Regression {

  import scala.math.log
  import breeze.linalg.{ DenseMatrix, DenseVector }

  def main(args: Array[String]): Unit = {

    // **********************************
    // Interactive session starts here

    val csv = CSV("data/regression.csv")
    val oi = csv("OI")
    val age = csv("Age")
    val sex = csv("Sex", x => if (x == "Male") 1.0 else 0.0)
    val y = oi map { log(_) }
    val X = modelMatrix(List(age, sex))
    val m = Lm(X, y)
    println(m)

    // Interactive session ends here
    // ***********************************

  }

  def modelMatrix(cols: List[DenseVector[Double]]): DenseMatrix[Double] = {
    //println("h1")
    val X = DenseMatrix.zeros[Double](cols(0).length, cols.length + 1)
    //println("h2")
    X(::, 0) := DenseVector.ones[Double](cols(0).length)
    //println("h3")
    for (i <- 1 to cols.length)
      X(::, i) := DenseVector(cols(i - 1).toArray)
    //println("h4")
    X
  }

}
