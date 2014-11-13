
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
    //println(oi)
    val age = getCol("Age", df2)
    //println(age)
    val sex = getColS("Sex", df2).mapValues(x => if (x == "Male") 1.0 else 0.0)
    //println(sex)
    val y = oi.mapValues { log(_) }
    //println(y)
    val x=age.joinPreserveColIx(sex)
    //println(x)
    println("h0")
    val X = ModelMatrix(x)
    println("h3")
    println(X.names)
    //println(X.X)
    println("h4")
    val m = Lm(X, y)
    println(m)

    // Interactive session ends here
    // ***********************************

  }

}
