/*
framian.scala

Test of "framian"

*/

import java.io.File
import scala.annotation.tailrec
import spire.implicits._
import framian.Index
import framian.csv.Csv

object FramianTest {

  def main(args: Array[String]) = {
    println("Hello")
    val df=Csv.parseFile(new File("my-df.csv")).labeled.toFrame
    println(df)
    println("Done")
  }

}



