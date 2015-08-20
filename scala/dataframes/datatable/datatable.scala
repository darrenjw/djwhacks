/*
datatable.scala

Test of "scala-datatable" and "scala-csv"

*/


object DatatableTest {

  def main(args: Array[String]) = {

import java.io.{File,FileReader}
import com.github.tototoshi.csv._
import com.github.martincooper.datatable._
import scala.annotation.tailrec
import scala.util.Try

    val reader=CSVReader.open(new FileReader("../r/cars93.csv"))
    val all=reader.allWithHeaders()
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
df.foreach(r=>println(r.values))

val df2=df.filter(row=>row.as[Double]("EngineSize")<=4.0).toDataTable
df2.foreach(r=>println(r.values))

val oldCol=df2.columns("Weight").as[Int]
val newCol=new DataColumn[Double]("WeightKG",oldCol.data.map{_.toDouble*0.453592})
val df3=df2.columns.add(newCol).get
df3.foreach(r=>println(r.values))


val writer = CSVWriter.open(new File("out.csv"))
writer.writeRow(df3.columns.map{_.name})
df3.foreach{r=>writer.writeRow(r.values)}
writer.close()

    println("Done")
  }



}




