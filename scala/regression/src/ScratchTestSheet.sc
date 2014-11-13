package regression

object ScratchTestSheet {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  import breeze.io.CSVReader
  import java.io.FileReader
  import breeze.stats.regression._
  import breeze.linalg._
  import org.saddle._
  import org.saddle.io._
  import FrameUtils._

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

  val dm = frame2mat(frame.mapValues(CsvParser.parseDouble))
                                                  //> Nov 13, 2014 10:51:30 AM com.github.fommil.jni.JniLoader liberalLoad
                                                  //| INFO: successfully loaded /tmp/jniloader7536542356624575954netlib-native_sys
                                                  //| tem-linux-x86_64.so
                                                  //| dm  : breeze.linalg.DenseMatrix[Double] = 5.0    65.0   NaN  
                                                  //| 3.75   40.0   NaN  
                                                  //| 7.6    52.0   NaN  
                                                  //| 2.45   45.0   NaN  
                                                  //| 5.4    72.0   NaN  
                                                  //| 10.7   64.0   NaN  
                                                  //| 6.15   67.0   NaN  
                                                  //| 5.15   66.0   NaN  
                                                  //| 2.15   47.0   NaN  
                                                  //| 2.45   44.0   NaN  
                                                  //| 5.94   46.0   NaN  
                                                  //| 9.94   52.0   NaN  
                                                  //| 6.59   55.0   NaN  
                                                  //| 9.35   64.0   NaN  
                                                  //| 5.15   62.0   NaN  
                                                  //| 7.09   37.0   NaN  
                                                  //| 8.44   39.0   NaN  
                                                  //| 2.34   45.0   NaN  
                                                  //| 2.65   39.0   NaN  
                                                  //| 3.34   54.0   NaN  
                                                  //| 5.8    37.0   NaN  
                                                  //| 0.98   34.0   NaN  
                                                  //| 9.8    48.0   NaN  
                                                  //| 6.25   66.0   NaN  
                                                  //| 6.69   39.0   NaN  
                                                  //| 5.9    47.0   NaN  
                                                  //| 4.75   39.0   NaN  
                                                  //| 7.69   48.0   NaN  
                                                  //| 6.84   57.0   NaN  
                                                  //| 4.84   55.0   NaN  
                                                  //| 6.75   61.0   NaN  
                                                  //| 8.35   29.0   NaN  
                                                  //| 3.75   2
                                                  //| Output exceeds cutoff limit.

  mat2frame(dm, Index(((0 until dm.rows).toArray map { _.toString })), Index("a", "b", "c"))
                                                  //> res0: org.saddle.Frame[String,String,Double] = [101 x 3]
                                                  //|              a       b  c 
                                                  //|        ------- ------- -- 
                                                  //|   0 ->  5.0000 65.0000 NA 
                                                  //|   1 ->  3.7500 40.0000 NA 
                                                  //|   2 ->  7.6000 52.0000 NA 
                                                  //|   3 ->  2.4500 45.0000 NA 
                                                  //|   4 ->  5.4000 72.0000 NA 
                                                  //| ...
                                                  //|  96 ->  8.8900 57.0000 NA 
                                                  //|  97 -> 16.5000 56.0000 NA 
                                                  //|  98 ->  4.6500 53.0000 NA 
                                                  //|  99 -> 13.5000 56.0000 NA 
                                                  //| 100 -> 16.1000 66.0000 NA 
                                                  //| 

  Frame(Mat(2, 2, Array(1, 2, 3, 4)), Index(0, 1), Index("a", "b"))
                                                  //> res1: org.saddle.Frame[Int,String,Int] = [2 x 2]
                                                  //|       a  b 
                                                  //|      -- -- 
                                                  //| 0 ->  1  2 
                                                  //| 1 ->  3  4 
                                                  //| 

}