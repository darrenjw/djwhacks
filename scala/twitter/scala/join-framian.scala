/*
join-framian.scala

Joining twitter data using scala with framian

*/

import java.io.{ File, FileWriter, BufferedWriter }
import framian.{ Frame, Index, Cols, Rows }
import framian.csv.{ Csv, CsvFormat, CsvRow }

object FramianJoin {

  def main(args: Array[String]) = {
    println("Hello")

    println("Reading and building mapper")
    val mapper = Csv.parseFile(new File("../data/mapping.csv")).labeled.toFrame
    val id = mapper.get(Cols("id").as[String]).values
    val sn = mapper.get(Cols("username").as[String]).values
    val snMap = (id zip sn).toMap
    //println(snMap)
    //println(mapper)

    println("Reading and parsing tweets")
    //val csv = Csv.parseFile(new File("../data/first1k.csv")).labeled.toFrame
    val csv = Csv.parseFile(new File("../data/tweets.csv")).labeled.toFrame

    println("Writing id list")
    val idList = csv.get(Cols("id").as[String]).values.toSet
    val out = new BufferedWriter(new FileWriter(new File("idlist.txt")))
    for (id <- idList) { out.write(id + "\n") }
    out.close

    println("Augmenting frame with additional column")
    val augmented = csv.map(Cols("id").as[String], "screenname")(x => snMap.getOrElse(x, ""))
    println(csv.colIndex)
    println("Done. Now writing to file...")
    println(augmented.colIndex)
    frame2csv(augmented, "augmented.csv")
    println("Done")
  }

  // BUG in this regarding element ordering, but don't think it's in my code...
  def frame2csv[Row, Col](df: Frame[Row, Col], filename: String): Unit = {
    val csv = Csv.fromFrame(new CsvFormat(",", header = true))(df)
    val out = new BufferedWriter(new FileWriter(new File(filename)))
    out.write(csv.toString)
    out.close
  }

}



