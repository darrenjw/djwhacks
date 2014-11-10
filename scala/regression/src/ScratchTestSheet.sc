package statslang

object ScratchTestSheet {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  import breeze.io.CSVReader
  import java.io.FileReader
  import breeze.stats.regression._
  import breeze.linalg._
  import org.saddle._
  import org.saddle.io._

  val csv = CSV("/home/ndjw1/src/git/statslang/scala/data/regression.csv")
                                                  //> csv  : statslang.CSV = statslang.CSV@578f7a8

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
 
                       val sex = csv("Sex", x => if (x == "Male") 1.0 else 0.0)
                                                  //> sex  : breeze.linalg.DenseVector[Double] = DenseVector(0.0, 0.0, 0.0, 0.0, 0
                                                  //| .0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
                                                  //| 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                                  //| , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  //|  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                                  //| 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1
                                                  //| .0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.
                                                  //| 0, 1.0, 1.0, 1.0, 1.0)
 
 import Regression.modelMatrix
 val X = modelMatrix(List(age, sex))              //> Nov 06, 2014 10:41:08 PM com.github.fommil.jni.JniLoader liberalLoad
                                                  //| INFO: successfully loaded /tmp/jniloader6772812544308432806netlib-native_sys
                                                  //| tem-linux-x86_64.so
                                                  //| X  : breeze.linalg.DenseMatrix[Double] = 1.0  65.0  0.0  
                                                  //| 1.0  40.0  0.0  
                                                  //| 1.0  52.0  0.0  
                                                  //| 1.0  45.0  0.0  
                                                  //| 1.0  72.0  0.0  
                                                  //| 1.0  64.0  0.0  
                                                  //| 1.0  67.0  0.0  
                                                  //| 1.0  66.0  0.0  
                                                  //| 1.0  47.0  0.0  
                                                  //| 1.0  44.0  0.0  
                                                  //| 1.0  46.0  0.0  
                                                  //| 1.0  52.0  0.0  
                                                  //| 1.0  55.0  0.0  
                                                  //| 1.0  64.0  0.0  
                                                  //| 1.0  62.0  0.0  
                                                  //| 1.0  37.0  0.0  
                                                  //| 1.0  39.0  0.0  
                                                  //| 1.0  45.0  0.0  
                                                  //| 1.0  39.0  0.0  
                                                  //| 1.0  54.0  0.0  
                                                  //| 1.0  37.0  0.0  
                                                  //| 1.0  34.0  0.0  
                                                  //| 1.0  48.0  0.0  
                                                  //| 1.0  66.0  0.0  
                                                  //| 1.0  39.0  0.0  
                                                  //| 1.0  47.0  0.0  
                                                  //| 1.0  39.0  0.0  
                                                  //| 1.0  48.0  0.0  
                                                  //| 1.0  57.0  0.0  
                                                  //| 1.0  55.0  0.0  
                                                  //| 1.0  61.0  0.0  
                                                  //| 1.0  29.0  0.0  
                                                  //| 1.0  27.0  0.0  
                                                  //| 1.0  62.0  0.0  
                                                  //| 1.0  37.0  0.0  
                                                  //| 1.0  63.0  0.0  
                                                  //| 1.0  58.0  0.0  
                                                  //| 
                                                  //| Output exceeds cutoff limit.

def thinQR(A: DenseMatrix[Double]): (DenseMatrix[Double],DenseMatrix[Double]) = {
     import breeze.linalg.qr
     import breeze.linalg.qr.QR
     val n=A.rows
     val p=A.cols
     val QR(_Q,_R)=qr(X)
     (_Q( :: , 0 until p),_R( 0 until p , :: ))
  }                                               //> thinQR: (A: breeze.linalg.DenseMatrix[Double])(breeze.linalg.DenseMatrix[Dou
                                                  //| ble], breeze.linalg.DenseMatrix[Double])
  

val myQR=thinQR(X)                                //> Nov 06, 2014 10:41:09 PM com.github.fommil.jni.JniLoader load
                                                  //| INFO: already loaded netlib-native_system-linux-x86_64.so
                                                  //| myQR  : (breeze.linalg.DenseMatrix[Double], breeze.linalg.DenseMatrix[Double
                                                  //| ]) = (-0.10000000000000009  0.11772887690882278    -0.07231996006200676    
                                                  //| -0.1                  -0.07252464104337977   -0.037585774206736165   
                                                  //| -0.1                  0.01879704757367766    -0.05425818341726615    
                                                  //| -0.1                  -0.03447393745293917   -0.04453261137779039    
                                                  //| -0.1                  0.17099986193544003    -0.08204553210148277    
                                                  //| -0.1                  0.11011873619073509    -0.07093059262779614    
                                                  //| -0.1                  0.13294915834499943    -0.07509869493042862    
                                                  //| -0.1                  0.12533901762691133    -0.07370932749621778    
                                                  //| -0.1                  -0.019253656016762932  -0.04731134624621204    
                                                  //| -0.1                  -0.04208407817102729   -0.04314324394357956    
                                                  //| -0.1                  -0.0268637967348510
                                                  //| Output exceeds cutoff limit.
val Q=myQR._1                                     //> Q  : breeze.linalg.DenseMatrix[Double] = -0.10000000000000009  0.11772887690
                                                  //| 882278    -0.07231996006200676    
                                                  //| -0.1                  -0.07252464104337977   -0.037585774206736165   
                                                  //| -0.1                  0.01879704757367766    -0.05425818341726615    
                                                  //| -0.1                  -0.03447393745293917   -0.04453261137779039    
                                                  //| -0.1                  0.17099986193544003    -0.08204553210148277    
                                                  //| -0.1                  0.11011873619073509    -0.07093059262779614    
                                                  //| -0.1                  0.13294915834499943    -0.07509869493042862    
                                                  //| -0.1                  0.12533901762691133    -0.07370932749621778    
                                                  //| -0.1                  -0.019253656016762932  -0.04731134624621204    
                                                  //| -0.1                  -0.04208407817102729   -0.04314324394357956    
                                                  //| -0.1                  -0.02686379673485105   -0.04592197881200122    
                                                  //| -0.1                  0.01879704757367766    -0.05425818341726619    
                                                  //| -0.1                  0.04162746972794201    -0.05842628571989
                                                  //| Output exceeds cutoff limit.
val R=myQR._2                                     //> R  : breeze.linalg.DenseMatrix[Double] = -10.0  -495.29999999999995  -2.0000
                                                  //| 000000000004  
                                                  //| 0.0    131.40361486656295   0.7183972837875185   
                                                  //| 0.0    0.0                  3.934959382591733    
Q.rows                                            //> res0: Int = 100
Q.cols                                            //> res1: Int = 3
R.rows                                            //> res2: Int = 3
R.cols                                            //> res3: Int = 3

Q.t * Q                                           //> res4: breeze.linalg.DenseMatrix[Double] = 1.0000000000000007       -7.979727
                                                  //| 989493313E-17  -2.0816681711721685E-16  
                                                  //| -7.979727989493313E-17   0.9999999999999992      -2.42861286636753E-17    
                                                  //| -2.0816681711721685E-16  -2.42861286636753E-17   0.9999999999999999       
 

val file=CsvFile("/home/ndjw1/src/git/statslang/scala/data/regression.csv")
                                                  //> file  : org.saddle.io.CsvFile = CsvFile(/home/ndjw1/src/git/statslang/scala/
                                                  //| data/regression.csv, encoding: UTF-8)

val frame=CsvParser.parse(file).withColIndex(0)   //> frame  : org.saddle.Frame[Int,String,String] = [101 x 3]
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
frame.col("OI")                                   //> res5: org.saddle.Frame[Int,String,String] = [101 x 1]
                                                  //|          OI 
                                                  //|        ---- 
                                                  //|   1 ->    5 
                                                  //|   2 -> 3.75 
                                                  //|   3 ->  7.6 
                                                  //|   4 -> 2.45 
                                                  //|   5 ->  5.4 
                                                  //| ...
                                                  //|  97 -> 8.89 
                                                  //|  98 -> 16.5 
                                                  //|  99 -> 4.65 
                                                  //| 100 -> 13.5 
                                                  //| 101 -> 16.1 
                                                  //| 

frame head(5)                                     //> res6: org.saddle.Frame[Int,String,String] = [5 x 3]
                                                  //|        OI Age    Sex 
                                                  //|      ---- --- ------ 
                                                  //| 1 ->    5  65 Female 
                                                  //| 2 -> 3.75  40 Female 
                                                  //| 3 ->  7.6  52 Female 
                                                  //| 4 -> 2.45  45 Female 
                                                  //| 5 ->  5.4  72 Female 
                                                  //| 
frame.colType[Int]                                //> res7: org.saddle.Frame[Int,String,Int] = Empty Frame

val p=Panel(Vec(1,2,3), Vec("a","b","c"))         //> p  : org.saddle.Frame[Int,Int,Any] = [3 x 2]
                                                  //|       0  1 
                                                  //|      -- -- 
                                                  //| 0 ->  1  a 
                                                  //| 1 ->  2  b 
                                                  //| 2 ->  3  c 
                                                  //| 
p.colType[Int]                                    //> res8: org.saddle.Frame[Int,Int,Int] = [3 x 1]
                                                  //|       0 
                                                  //|      -- 
                                                  //| 0 ->  1 
                                                  //| 1 ->  2 
                                                  //| 2 ->  3 
                                                  //| 
p.colType[Int,String]                             //> res9: org.saddle.Frame[Int,Int,Any] = [3 x 2]
                                                  //|       0  1 
                                                  //|      -- -- 
                                                  //| 0 ->  1  a 
                                                  //| 1 ->  2  b 
                                                  //| 2 ->  3  c 
                                                  //| 
p.colType[String]                                 //> res10: org.saddle.Frame[Int,Int,String] = [3 x 1]
                                                  //|       1 
                                                  //|      -- 
                                                  //| 0 ->  a 
                                                  //| 1 ->  b 
                                                  //| 2 ->  c 
                                                  //| 
//Panel(frame.col("OI"),frame.col("Age"),frame.col("Sex"))
val M=mat.randn(10,4)                             //> M  : org.saddle.Mat[Double] = [10 x 4]
                                                  //|  0.6438 -1.4008 -0.0863  0.0562 
                                                  //| -0.0186  0.4583  0.1704 -0.3647 
                                                  //|  1.1427 -0.8928 -0.9997  0.3752 
                                                  //| -1.1355  0.2442 -1.0852  0.2601 
                                                  //| ...
                                                  //|  1.3131 -1.2620 -0.5415 -1.0106 
                                                  //|  0.6582 -0.4368  0.1244  1.4828 
                                                  //| -1.9215 -1.1584 -0.5987 -0.1992 
                                                  //|  0.7840  0.0206  0.7061 -0.6931 
                                                  //| 

    
0 until 3                                         //> res11: scala.collection.immutable.Range = Range(0, 1, 2)


}