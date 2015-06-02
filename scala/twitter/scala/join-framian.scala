/*
join.scala

Joining twitter data using scala with framian

*/

import java.io.File
import framian.{Index,Cols}
import framian.csv.Csv

object FramianTest {

  def main(args: Array[String]) = {
    println("Hello")
    val mapper=Csv.parseFile(new File("../data/mapping.csv")).labeled.toFrame
    val id=mapper.get(Cols("id").as[String]).values
    val sn=mapper.get(Cols("username").as[String]).values
    val snMap=(id zip sn).toMap
    //println(snMap)
    //println(mapper)
    val csv=Csv.parseFile(new File("../data/first1k.csv")).labeled.toFrame
    val augmented=csv.map(Cols("id").as[String],"screenname")(x=>snMap.getOrElse(x,""))
    //println(csv)
    println("Done")
  }

}



