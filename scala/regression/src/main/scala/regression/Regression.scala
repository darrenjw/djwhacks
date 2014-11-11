
package statslang

object Regression {

  import scala.math.log
  import breeze.linalg.{ DenseMatrix, DenseVector }
  import org.saddle._
  import org.saddle.io._

  def main(args: Array[String]): Unit = {

    // **********************************
    // Interactive session starts here

    val file = CsvFile("data/regression.csv")
    val df = CsvParser.parse(file).withColIndex(0)

    val df2 = df.row((getCol("Age", df).filter(_ > 0.0)).index.toVec)
    val oi = getCol("OI", df2)
    val age = getCol("Age", df2)
    val sex = getColS("Sex", df2).mapValues(x => if (x == "Male") 1.0 else 0.0)

    val y = DenseVector[Double](oi.mapValues { log(_) }.values.contents)
    val X = modelMatrix(List(age, sex))
    val m = Lm(X, y)
    println(m)

    // Interactive session ends here
    // ***********************************

  }

  def getCol(colName: String, sdf: Frame[Int, String, String]): Series[Int, Double] = {
    sdf.firstCol(colName).mapValues(CsvParser.parseDouble)
  }

  def getColS(colName: String, sdf: Frame[Int, String, String]): Series[Int, String] = {
    sdf.firstCol(colName)
  }

  def modelMatrix(cols: List[Series[Int, Double]]): DenseMatrix[Double] = {
    //println("h1")
    val X = DenseMatrix.zeros[Double](cols(0).length, cols.length + 1)
    //println("h2")
    X(::, 0) := DenseVector.ones[Double](cols(0).length)
    //println("h3")
    for (i <- 1 to cols.length)
      X(::, i) := DenseVector(cols(i - 1).values.contents)
    //println("h4")
    X
  }

}
