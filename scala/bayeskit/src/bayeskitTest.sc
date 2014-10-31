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
  System.getProperty("user.dir")                  //> res6: String = /home/ndjw1/src/git/djwhacks/scala/bayeskit

  Source.fromFile("LVpreyNoise10.txt").getLines   //> res7: Iterator[String] = non-empty iterator

  0 to 30                                         //> res8: scala.collection.immutable.Range.Inclusive = Range(0, 1, 2, 3, 4, 5, 6
                                                  //| , 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 2
                                                  //| 6, 27, 28, 29, 30)

}