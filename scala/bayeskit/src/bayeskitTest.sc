object bayeskitTest {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  import bayeskit.bayeskit._

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

  val state = stepLV(new State(100, 50), 0, 10, Vector(1.0, 0.005, 0.6))
                                                  //> state  : bayeskit.bayeskit.State = (0,16)

  def fun(a: Int): (Int => Int) = {
    val c = a + 1
    (b: Int) =>
      {
        c + b
      }
  }                                               //> fun: (a: Int)Int => Int

  fun(1)(2)                                       //> res5: Int = 4

  List(1, 2, 3) zip List(2, 3)                    //> res6: List[(Int, Int)] = List((1,2), (2,3))

  def diff(l: List[Int]): List[Int] = {
    (l.tail zip l) map { x => x._1 - x._2 }
  }                                               //> diff: (l: List[Int])List[Int]

  diff(List(1, 2, 3, 4, 6, 7, 9))                 //> res7: List[Int] = List(1, 1, 1, 2, 1, 2)

  Vector[Double](1.0,2.0)                         //> res8: scala.collection.immutable.Vector[Double] = Vector(1.0, 2.0)
  




}