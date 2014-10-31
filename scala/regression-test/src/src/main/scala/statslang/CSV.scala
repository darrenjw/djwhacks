package statslang

import breeze.io.CSVReader
import java.io.FileReader
import scala.collection.mutable.HashMap
import breeze.linalg.DenseVector

class CSV(head: IndexedSeq[String], body: IndexedSeq[IndexedSeq[String]]) {

  val fields = head.toList
  val pairs = (0 until head.length) zip head
  val map = pairs.map(x => x._2 -> x._1).toMap

  def apply(field: String): DenseVector[Double] = {
    gets(field) map { _.toDouble }
  }

  def apply(field: String, conv: String => Double): DenseVector[Double] = {
    gets(field) map { conv(_) }
  }

  def gets(field: String): DenseVector[String] = {
    val idx = map(field)
    val seq = body map { _(idx) }
    new DenseVector[String](seq.toArray)
  }

}

object CSV {

  def apply(filename: String): CSV = {
    val csv = CSVReader.read(new FileReader(filename))
    val header = csv(0)
    val rest = csv.slice(1, csv.length) filter { _(1).toDouble > 0 }
    new CSV(header, rest)
  }

}
