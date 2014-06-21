object bayeskitTest {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  // shift control b to re evaluate the sheet

  import bayeskit.bayeskit._

  import breeze.stats.distributions._
  import bayeskit.sim._
  import bayeskit.lvsim.stepLV
  import bayeskit.pfilter._
  import bayeskit.pmmh._

  import org.apache.commons.math3.distribution._

  import breeze.stats.distributions._
  // import breeze.linalg._

  1 + 2                                           //> res0: Int(3) = 3

  0 to 9                                          //> res1: scala.collection.immutable.Range.Inclusive = Range(0, 1, 2, 3, 4, 5, 6
                                                  //| , 7, 8, 9)
  //linspace(1.0,10.0,20)

  //val cat=new EnumeratedIntegerDistribution((0 to 9).toArray, linspace(0.0,1.0,10).toArray)
  //cat.sample

  1 + 1                                           //> res2: Int(2) = 2
  val h = (1, 2, 3)                               //> h  : (Int, Int, Int) = (1,2,3)
  h._1                                            //> res3: Int = 1

  (1, "b")                                        //> res4: (Int, String) = (1,b)
  val a = List(1, 2, 3)                           //> a  : List[Int] = List(1, 2, 3)
  val b = List("a", "b", "c")                     //> b  : List[String] = List(a, b, c)
  val c = a zip b                                 //> c  : List[(Int, String)] = List((1,a), (2,b), (3,c))

  Vector(1, 2, 3).map { _ * 2 }                   //> res5: scala.collection.immutable.Vector[Int] = Vector(2, 4, 6)

  // read a data file
  import scala.io.Source
  System.getProperty("user.dir")                  //> res6: String = /home/ndjw1/Applications/eclipse

  Source.fromFile("LVpreyNoise10.txt").getLines   //> java.io.FileNotFoundException: LVpreyNoise10.txt (No such file or directory)
                                                  //| 
                                                  //| 	at java.io.FileInputStream.open(Native Method)
                                                  //| 	at java.io.FileInputStream.<init>(FileInputStream.java:146)
                                                  //| 	at scala.io.Source$.fromFile(Source.scala:90)
                                                  //| 	at scala.io.Source$.fromFile(Source.scala:75)
                                                  //| 	at scala.io.Source$.fromFile(Source.scala:53)
                                                  //| 	at bayeskitTest$$anonfun$main$1.apply$mcV$sp(bayeskitTest.scala:42)
                                                  //| 	at org.scalaide.worksheet.runtime.library.WorksheetSupport$$anonfun$$exe
                                                  //| cute$1.apply$mcV$sp(WorksheetSupport.scala:76)
                                                  //| 	at org.scalaide.worksheet.runtime.library.WorksheetSupport$.redirected(W
                                                  //| orksheetSupport.scala:65)
                                                  //| 	at org.scalaide.worksheet.runtime.library.WorksheetSupport$.$execute(Wor
                                                  //| ksheetSupport.scala:75)
                                                  //| 	at bayeskitTest$.main(bayeskitTest.scala:1)
                                                  //| 	at bayeskitTest.main(bayeskitTest.scala)

  0 to 30

}