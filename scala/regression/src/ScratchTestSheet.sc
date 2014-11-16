package regression

object ScratchTestSheet {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  import breeze.io.CSVReader
  import java.io.FileReader
  import breeze.stats.regression._
  import breeze.linalg._
  import breeze.plot._
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
                                                  //> Nov 16, 2014 9:18:52 PM com.github.fommil.jni.JniLoader liberalLoad
                                                  //| INFO: successfully loaded /tmp/jniloader497204747364478781netlib-native_syst
                                                  //| em-linux-x86_64.so
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
                                                  //| 3.75   27.
                                                  //| Output exceeds cutoff limit.

  val f = Figure()                                //> f  : breeze.plot.Figure = breeze.plot.Figure@47cd0dbb
  val p = f.subplot(0)                            //> p  : breeze.plot.Plot = breeze.plot.Plot@66bfeeae
  p += plot(dm(::, 1), dm(::, 0), '.',"red")      //> res0: breeze.plot.Plot = breeze.plot.Plot@66bfeeae/

}