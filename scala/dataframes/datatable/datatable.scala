/*
datatable.scala

Test of "scala-datatable" and "scala-csv"

*/

import java.io.FileReader
import com.github.tototoshi.csv._
import com.github.martincooper.datatable._
import scala.annotation.tailrec
import scala.util.Try


object DatatableTest {

  def main(args: Array[String]) = {
    println("Hello")
    println("First some messing around with a CSV file using scala-csv")
    val reader=CSVReader.open(new FileReader("../r/cars93.csv"))
    val all=reader.allWithHeaders()
    println(all)
    reader.close()

object StringCol

val colTypes=Map("DriveTrain" -> StringCol, "Min.Price" -> Double, "Cylinders" -> Int, "Horsepower" -> Int, "Length" -> Int, "Make" -> StringCol, "Passengers" -> Int, "Width" -> Int, "Fuel.tank.capacity" -> Double, "Origin" -> StringCol, "Wheelbase" -> Int, "Price" -> Double, "Luggage.room" -> Double, "Weight" -> Int, "Model" -> StringCol, "Max.Price" -> Double, "Manufacturer" -> StringCol, "EngineSize" -> Double, "AirBags" -> StringCol, "Man.trans.avail" -> StringCol, "Rear.seat.room" -> Double, "RPM" -> Int, "Turn.circle" -> Double, "MPG.highway" -> Int, "MPG.city" -> Int, "Rev.per.mile" -> Int, "Type" -> StringCol)

val ks=colTypes.keys
val colSet=ks map {key => (key,all map {row => row(key)}) }
val dataCols=colSet map {pair => colTypes(pair._1) match { 
  case StringCol => new DataColumn[String](pair._1,pair._2)
  case Int       => new DataColumn[Int](pair._1,pair._2 map {x=>
                                        Try(x.toInt).toOption.getOrElse(-99)})
  case Double    => new DataColumn[Double](pair._1,pair._2 map {x=>
                                        Try(x.toDouble).toOption.getOrElse(-99.0)})
  } 
}
val df=DataTable("Cars93",dataCols).get

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
    // filter the frame and convert the view back to a table
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




