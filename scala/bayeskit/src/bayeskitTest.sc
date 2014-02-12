object bayeskitTest {
  println("Welcome to the Scala worksheet")

  import bayeskit.bayeskit._

  import org.apache.commons.math3.distribution._
  
  import breeze.stats.distributions._
  // import breeze.linalg._
  
  1 + 2

  0 to 9
  //linspace(1.0,10.0,20)
  
 
  //val cat=new EnumeratedIntegerDistribution((0 to 9).toArray, linspace(0.0,1.0,10).toArray)
  //cat.sample
  

  1 + 1
  val h = (1, 2, 3)
  h._1

  (1, "b")
  val a = List(1, 2, 3)
  val b = List("a", "b", "c")
  val c = a zip b

  val state = stepLV(new State(100, 50), 0, 10, Vector(1.0, 0.005, 0.6))

  def fun(a: Int): (Int => Int) = {
    val c = a + 1
    (b: Int) =>
      {
        c + b
      }
  }

  fun(1)(2)

  List(1, 2, 3) zip List(2, 3)

  def diff(l: List[Int]): List[Int] = {
    (l.tail zip l) map { x => x._1 - x._2 }
  }

  diff(List(1, 2, 3, 4, 6, 7, 9))

  Vector[Double](1.0,2.0)
  




}