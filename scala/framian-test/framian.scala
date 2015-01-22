/*
framian.scala

Test of "framian"

*/

import java.io.File
import framian.{Index,Cols}
import framian.csv.Csv

object FramianTest {

  def main(args: Array[String]) = {
    println("Hello")
    val df=Csv.parseFile(new File("my-df.csv")).labeled.toFrame
    println(""+df.rows+" "+df.cols)
    val df2=df.filter(Cols("Age").as[Int])( _ > 0 )
    println(""+df2.rows+" "+df2.cols)
    println(df2)
    println(df2.get(Cols("OI").as[Double]).values)
    println(df2.get(Cols("Age").as[Int]).values)
    println(df2.get(Cols("Sex").as[String]).values)
    println(""+df2.rows+" "+df2.cols)
    println(df2.colIndex)
    println("Done")
  }

}



