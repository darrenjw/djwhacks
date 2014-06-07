object bayeskitTest {
  println("Welcome to the Scala worksheet")

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

  val state = stepLV(Vector(100, 50), 0, 10, Vector(1.0, 0.005, 0.6))

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

  Vector[Double](1.0, 2.0)

  def simPrior(n: Int, t: Time, th: Parameter): Vector[State] = {
    val prey = new Poisson(100.0).sample(n).toVector
    val predator = new Poisson(50.0).sample(n).toVector
    prey.zip(predator) map { x => Vector(x._1, x._2) }
  }
  def obsLik(s: State, o: Observation, th: Parameter): Double = {
    new Gaussian(s(0), 10.0).pdf(o(0))
  }
  val truth = simTs(Vector(100, 50), 0, 30, 2.0, stepLV, Vector(1.0, 0.005, 0.6))
  val data = truth map { x => (x._1, Vector(x._2(0).toDouble)) }
  val mll = pfMLLik(100, simPrior, 0.0, stepLV, obsLik, data)
  val mllSample = mll(Vector(1.0, 0.005, 0.6))

  val pmll = pfMLLikPar(100, simPrior, 0.0, stepLV, obsLik, data)
  val pmllSample = pmll(Vector(1.0, 0.005, 0.6))

  mll(Vector(1.0, 0.005, 0.6))
  mll(Vector(1.0, 0.005, 0.6))
  mll(Vector(1.0, 0.005, 0.6))

  pmll(Vector(1.0, 0.005, 0.6))
  pmll(Vector(1.0, 0.005, 0.6))
  pmll(Vector(1.0, 0.005, 0.6))

  Vector(1,2,3).map{_*2}

 // read a data file
 import scala.io.Source
 System.getProperty("user.dir")

 Source.fromFile("LVpreyNoise10.txt").getLines
 
 
  0 to 30
  
  
}