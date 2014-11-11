package statslang

object ScratchTestSheet {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  import breeze.io.CSVReader
  import java.io.FileReader
  import breeze.stats.regression._
  import breeze.linalg._
  import org.saddle._
  import org.saddle.io._

  val file = CsvFile("/home/ndjw1/src/git/djwhacks/scala/regression/data/regression.csv")
                                                  //> file  : org.saddle.io.CsvFile = CsvFile(/home/ndjw1/src/git/djwhacks/scala/r
                                                  //| egression/data/regression.csv, encoding: UTF-8)

  val frame = CsvParser.parse(file).withColIndex(0)
                                                  //> frame  : org.saddle.Frame[Int,String,String] = [101 x 3]
                                                  //|          OI Age    Sex 
                                                  //|        ---- --- ------ 
                                                  //|   1 ->    5  65 Female 
                                                  //|   2 -> 3.75  40 Female 
                                                  //|   3 ->  7.6  52 Female 
                                                  //|   4 -> 2.45  45 Female 
                                                  //|   5 ->  5.4  72 Female 
                                                  //| ...
                                                  //|  97 -> 8.89  57   Male 
                                                  //|  98 -> 16.5  56   Male 
                                                  //|  99 -> 4.65  53   Male 
                                                  //| 100 -> 13.5  56   Male 
                                                  //| 101 -> 16.1  66   Male 
                                                  //| 
  frame.toMat                                     //> res0: org.saddle.Mat[String] = [101 x 3]
                                                  //|    5 65 Female 
                                                  //| 3.75 40 Female 
                                                  //|  7.6 52 Female 
                                                  //| 2.45 45 Female 
                                                  //| ...
                                                  //| 16.5 56   Male 
                                                  //| 4.65 53   Male 
                                                  //| 13.5 56   Male 
                                                  //| 16.1 66   Male 
                                                  //| 
  frame.mapValues(CsvParser.parseDouble)          //> res1: org.saddle.Frame[Int,String,Double] = [101 x 3]
                                                  //|             OI     Age Sex 
                                                  //|        ------- ------- --- 
                                                  //|   1 ->  5.0000 65.0000  NA 
                                                  //|   2 ->  3.7500 40.0000  NA 
                                                  //|   3 ->  7.6000 52.0000  NA 
                                                  //|   4 ->  2.4500 45.0000  NA 
                                                  //|   5 ->  5.4000 72.0000  NA 
                                                  //| ...
                                                  //|  97 ->  8.8900 57.0000  NA 
                                                  //|  98 -> 16.5000 56.0000  NA 
                                                  //|  99 ->  4.6500 53.0000  NA 
                                                  //| 100 -> 13.5000 56.0000  NA 
                                                  //| 101 -> 16.1000 66.0000  NA 
                                                  //| 
frame.firstCol("OI").mapValues(CsvParser.parseDouble).index.toVec
                                                  //> res2: org.saddle.Vec[Int] = [101 x 1]
                                                  //|   1
                                                  //|   2
                                                  //|   3
                                                  //|   4
                                                  //|   5
                                                  //|  ... 
                                                  //|  97
                                                  //|  98
                                                  //|  99
                                                  //| 100
                                                  //| 101
                                                  //| 



  frame.firstCol("OI")                            //> res3: org.saddle.Series[Int,String] = [101 x 1]
                                                  //|   1 ->    5
                                                  //|   2 -> 3.75
                                                  //|   3 ->  7.6
                                                  //|   4 -> 2.45
                                                  //|   5 ->  5.4
                                                  //|  ... 
                                                  //|  97 -> 8.89
                                                  //|  98 -> 16.5
                                                  //|  99 -> 4.65
                                                  //| 100 -> 13.5
                                                  //| 101 -> 16.1
                                                  //| 

  frame head (5)                                  //> res4: org.saddle.Frame[Int,String,String] = [5 x 3]
                                                  //|        OI Age    Sex 
                                                  //|      ---- --- ------ 
                                                  //| 1 ->    5  65 Female 
                                                  //| 2 -> 3.75  40 Female 
                                                  //| 3 ->  7.6  52 Female 
                                                  //| 4 -> 2.45  45 Female 
                                                  //| 5 ->  5.4  72 Female 
                                                  //| 
  frame.colType[Int]                              //> res5: org.saddle.Frame[Int,String,Int] = Empty Frame

  val p = Panel(Vec(1, 2, 3), Vec("a", "b", "c")) //> p  : org.saddle.Frame[Int,Int,Any] = [3 x 2]
                                                  //|       0  1 
                                                  //|      -- -- 
                                                  //| 0 ->  1  a 
                                                  //| 1 ->  2  b 
                                                  //| 2 ->  3  c 
                                                  //| 
  p.colType[Int]                                  //> res6: org.saddle.Frame[Int,Int,Int] = [3 x 1]
                                                  //|       0 
                                                  //|      -- 
                                                  //| 0 ->  1 
                                                  //| 1 ->  2 
                                                  //| 2 ->  3 
                                                  //| 
  p.colType[Int, String]                          //> res7: org.saddle.Frame[Int,Int,Any] = [3 x 2]
                                                  //|       0  1 
                                                  //|      -- -- 
                                                  //| 0 ->  1  a 
                                                  //| 1 ->  2  b 
                                                  //| 2 ->  3  c 
                                                  //| 
  p.colType[String]                               //> res8: org.saddle.Frame[Int,Int,String] = [3 x 1]
                                                  //|       1 
                                                  //|      -- 
                                                  //| 0 ->  a 
                                                  //| 1 ->  b 
                                                  //| 2 ->  c 
                                                  //| 
  //Panel(frame.col("OI"),frame.col("Age"),frame.col("Sex"))
  val M = mat.randn(10, 4)                        //> M  : org.saddle.Mat[Double] = [10 x 4]
                                                  //|  0.0027  1.9962 -0.3900 -1.1164 
                                                  //| -0.6727 -1.0580  0.9872 -1.0680 
                                                  //|  0.6197  1.7411 -0.3499  0.3772 
                                                  //| -1.6588 -0.1323  0.0704  0.8569 
                                                  //| ...
                                                  //| -1.2386  0.4922 -0.3405  1.2820 
                                                  //| -0.2214 -0.3245 -1.0431  0.1615 
                                                  //|  1.0410 -1.9612 -0.1855 -0.0290 
                                                  //|  0.4434  1.1103 -0.8757  1.7793 
                                                  //| 

  0 until 3                                       //> res9: scala.collection.immutable.Range = Range(0, 1, 2)

}