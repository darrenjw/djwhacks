/*
datatable.scala

Test of "scala-datatable" and "scala-csv"

*/

import java.io.FileReader
import com.github.tototoshi.csv._
import scala.annotation.tailrec

object DatatableTest {

  def main(args: Array[String]) = {
    println("Hello")
    val reader=CSVReader.open(new FileReader("my-df.csv"))
    val all=reader.all()
    println(all)
    reader.close()
    println("Done")
  }



}




