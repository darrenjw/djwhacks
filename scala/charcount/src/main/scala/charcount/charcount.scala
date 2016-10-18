/*
charcount.scala
Counting characters in imperative and functional ways...

 */

import scala.io.Source
//import cats._
import cats.implicits._

object CharCount {

  val wordFile = "/usr/share/dict/words"
  val wordsString = Source.fromFile(wordFile).getLines.map(_.trim).map(_.toLowerCase).mkString
  val fullCharArray = wordsString.toCharArray.filter(_ >= ' ').filter(_!='\'')
  val shortCharArray = fullCharArray.take(500)


  def main(args: Array[String]): Unit = {
    println("hi")

    println(shortCharArray.toList.mkString)
    val fullCharArrayPar=fullCharArray.par
    val freqMap=fullCharArray.map(c => Map(c->1L)).reduce(_ |+| _)
    val freqMapP=fullCharArrayPar.map(c => Map(c->1L)).reduce(_ |+| _)
    println(freqMap)
    println(freqMapP)

    println("bye")
  }

}

// eof

