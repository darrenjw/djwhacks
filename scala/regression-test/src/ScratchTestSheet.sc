package statslang

object ScratchTestSheet {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  import breeze.io.CSVReader
  import java.io.FileReader
  import breeze.stats.regression._
  import breeze.linalg._

  val csv = CSV("/home/ndjw1/src/git/statslang/scala/data/regression.csv")
                                                  //> csv  : statslang.CSV = statslang.CSV@3b08a056

val head=csv.fields                               //> head  : List[String] = List(OI, Age, Sex)

  val age = csv("Age")                            //> age  : breeze.linalg.DenseVector[String] = DenseVector(65, 40, 52, 45, 72, 6
                                                  //| 4, 67, 66, 47, 44, 46, 52, 55, 64, 62, 37, 39, 45, 39, 54, 37, 34, 48, 66, 3
                                                  //| 9, 47, 39, 48, 57, 55, 61, 29, 27, 62, 37, 63, 58, 49, 29, 15, 35, 60, 37, 2
                                                  //| 6, 60, 40, 31, 41, 30, 56, 46, 65, 26, 52, 65, 64, 61, 47, 52, 54, 14, 54, 2
                                                  //| 5, 58, 52, 45, 48, 46, 73, 43, 56, 61, 54, 62, 45, 59, 54, 26, 34, 56, 69, 6
                                                  //| 5, 25, 53, 39, 67, 53, 64, 48, 62, 32, 51, 46, 56, 67, 57, 56, 53, 56, 66)

}