package statslang

object ScratchTestSheet {
  println("Welcome to the Scala worksheet")

  import breeze.io.CSVReader
  import java.io.FileReader
  import breeze.stats.regression._
  import breeze.linalg._
  import org.saddle._
  import org.saddle.io._

  val file = CsvFile("/home/ndjw1/src/git/djwhacks/scala/regression/data/regression.csv")

  val frame = CsvParser.parse(file).withColIndex(0)
  frame.toMat.contents

  frame.mapValues(CsvParser.parseDouble).toMat.contents

  DenseMatrix(2, 2, Array(1, 2, 3, 4))

}