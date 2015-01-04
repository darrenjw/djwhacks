/*
datatable.scala

Test of "scala-datatable" and "scala-csv"

*/

import java.io.FileReader
import com.github.tototoshi.csv._
import com.github.martincooper.datatable._
import scala.annotation.tailrec

object DatatableTest {

  def main(args: Array[String]) = {
    println("Hello")
    println("First some messing around with a CSV file using scala-csv")
    val reader=CSVReader.open(new FileReader("my-df.csv"))
    val all=reader.allWithHeaders()
    println(all)
    reader.close()
    val OI=all map {_("OI").toDouble}
    println(OI)
    val Age=all map {_("Age").toInt}
    println(Age)
    val Sex=all map {_("Sex")}
    println(Sex)
    println("Now some messing around with scala-datatable")
    val OICol=new DataColumn[Double]("OI",OI)
    val AgeCol=new DataColumn[Int]("Age",Age)
    val SexCol=new DataColumn[String]("Sex",Sex)
    val dtOpt=DataTable("MyDT",Seq(OICol,AgeCol,SexCol))
    val dt=dtOpt.get
    println(dt)
    val dtF=dt.filter(row => row.as[Int]("Age") > 0 ).toDataTable
    println(dtF)
    println(dtF.length)
    println(dtF.columns("Age").as[Int].data)
    println(dtF.columns(0).name)
    println(dtF.columns.as[Int]("Age").data)
    println(dtF.columns.as[Int]("Age").data.map{_.toDouble})
    println("Done")
  }



}




