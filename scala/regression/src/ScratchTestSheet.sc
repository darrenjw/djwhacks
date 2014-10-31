package statslang

object ScratchTestSheet {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  import breeze.io.CSVReader
  import java.io.FileReader
  import breeze.stats.regression._
  import breeze.linalg._

  val csv = CSV("/home/ndjw1/src/git/statslang/scala/data/regression.csv")
                                                  //> csv  : statslang.CSV = statslang.CSV@6b10bc5c

val head=csv.fields                               //> head  : List[String] = List(OI, Age, Sex)

  val age = csv("Age")                            //> age  : breeze.linalg.DenseVector[Double] = DenseVector(65.0, 40.0, 52.0, 45.
                                                  //| 0, 72.0, 64.0, 67.0, 66.0, 47.0, 44.0, 46.0, 52.0, 55.0, 64.0, 62.0, 37.0, 3
                                                  //| 9.0, 45.0, 39.0, 54.0, 37.0, 34.0, 48.0, 66.0, 39.0, 47.0, 39.0, 48.0, 57.0,
                                                  //|  55.0, 61.0, 29.0, 27.0, 62.0, 37.0, 63.0, 58.0, 49.0, 29.0, 15.0, 35.0, 60.
                                                  //| 0, 37.0, 26.0, 60.0, 40.0, 31.0, 41.0, 30.0, 56.0, 46.0, 65.0, 26.0, 52.0, 6
                                                  //| 5.0, 64.0, 61.0, 47.0, 52.0, 54.0, 14.0, 54.0, 25.0, 58.0, 52.0, 45.0, 48.0,
                                                  //|  46.0, 73.0, 43.0, 56.0, 61.0, 54.0, 62.0, 45.0, 59.0, 54.0, 26.0, 34.0, 56.
                                                  //| 0, 69.0, 65.0, 25.0, 53.0, 39.0, 67.0, 53.0, 64.0, 48.0, 62.0, 32.0, 51.0, 4
                                                  //| 6.0, 56.0, 67.0, 57.0, 56.0, 53.0, 56.0, 66.0)

}