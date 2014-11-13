
package regression

object Regression {

  import scala.math.log
  import breeze.linalg.{ DenseMatrix, DenseVector }
  import org.saddle._
  import org.saddle.io._
  import FrameUtils._

  def main(args: Array[String]): Unit = {

    // **********************************
    // Interactive session starts here

    val file = CsvFile("data/regression.csv")
    val df = CsvParser.parse(file).withColIndex(0)

    val df2 = df.row((getCol("Age", df).rfilter(_.at(0).get > 0.0)).rowIx.toVec)
    println(df2)
    val oi = getCol("OI", df2)
    val age = getCol("Age", df2)
    val sex = getColS("Sex", df2).mapValues(x => if (x == "Male") 1.0 else 0.0)
    val y = oi.mapValues { log(_) }
    val x = age.joinPreserveColIx(sex)
    val X = ModelMatrix(x)
    val m = Lm(X, y)
    println(m)

    // Interactive session ends here
    // ***********************************

  }

}
