/*
join-framian.scala

Joining twitter data using scala with framian

*/

import java.io.{File,FileWriter,BufferedWriter}
import framian.{Frame,Index,Cols,Rows}
import framian.csv.Csv

object FramianJoin {

  def main(args: Array[String]) = {
    println("Hello")

    println("Reading and building mapper")
    val mapper=Csv.parseFile(new File("../data/mapping.csv")).labeled.toFrame
    val id=mapper.get(Cols("id").as[String]).values
    val sn=mapper.get(Cols("username").as[String]).values
    val snMap=(id zip sn).toMap
    //println(snMap)
    //println(mapper)

    println("Reading and parsing tweets")
    val csv=Csv.parseFile(new File("../data/first1k.csv")).labeled.toFrame

    println("Writing id list")
    val idList=csv.get(Cols("id").as[String]).values.toSet
    val out=new BufferedWriter(new FileWriter(new File("idlist.txt")))
    for (id <-idList) {out.write(id+"\n")}
    out.close

    println("Augmenting frame with additional column")
    val augmented=csv.map(Cols("id").as[String],"screenname")(x=>snMap.getOrElse(x,""))
    //println(csv)
    println("Done. Now writing to file - very slow...")
    frame2csv(augmented,"augmented.csv")
    println("Done")
  }



  def frame2csv[Row,Col](df: Frame[Row,Col],filename: String): Unit = {
    import com.github.tototoshi.csv._
    val writer = CSVWriter.open(filename)
    writer.writeRow(df.colIndex.map{_._1}.toList)
    for (rowIdx <- df.rowIndex) {
      val row=df.get(Rows(rowIdx._1)).values.flatMap{_.values}.map{_._2}.map{_.get}
      writer.writeRow(row)
    }
    writer.close
  }

}



