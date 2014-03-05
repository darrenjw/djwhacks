object bayeskitTest {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  import bayeskit.bayeskit._

  import breeze.stats.distributions._
  import bayeskit.sim._
  import bayeskit.lvsim.stepLV
  import bayeskit.pfilter._

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

  val state = stepLV(Vector(100, 50), 0, 10, Vector(1.0, 0.005, 0.6))
                                                  //> state  : bayeskit.sim.State = Vector(48, 18)

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

  Vector[Double](1.0, 2.0)                        //> res8: scala.collection.immutable.Vector[Double] = Vector(1.0, 2.0)

  def simPrior(n: Int, t: Time, th: Parameter): Vector[State] = {
    val prey = new Poisson(100.0).sample(n).toVector
    val predator = new Poisson(50.0).sample(n).toVector
    prey.zip(predator) map { x => Vector(x._1, x._2) }
  }                                               //> simPrior: (n: Int, t: bayeskit.sim.Time, th: bayeskit.sim.Parameter)Vector[
                                                  //| bayeskit.sim.State]
  def obsLik(s: State, o: Observation, th: Parameter): Double = {
    new Gaussian(s(0), 10.0).pdf(o(0))
  }                                               //> obsLik: (s: bayeskit.sim.State, o: bayeskit.sim.Observation, th: bayeskit.s
                                                  //| im.Parameter)Double
  val truth = simTs(Vector(100, 50), 0, 30, 2.0, stepLV, Vector(1.0, 0.005, 0.6))
                                                  //> truth  : bayeskit.sim.StateTS = List((0.0,Vector(100, 50)), (2.0,Vector(391
                                                  //| , 128)), (4.0,Vector(50, 557)), (6.0,Vector(13, 218)), (8.0,Vector(28, 70))
                                                  //| , (10.0,Vector(173, 35)), (12.0,Vector(394, 336)), (14.0,Vector(22, 359)), 
                                                  //| (16.0,Vector(12, 128)), (18.0,Vector(57, 41)), (20.0,Vector(280, 56)), (22.
                                                  //| 0,Vector(179, 518)), (24.0,Vector(9, 284)), (26.0,Vector(8, 105)), (28.0,Ve
                                                  //| ctor(75, 47)), (30.0,Vector(364, 111)))
  val data = truth map { x => (x._1, Vector(x._2(0).toDouble)) }
                                                  //> data  : List[(bayeskit.sim.Time, scala.collection.immutable.Vector[Double])
                                                  //| ] = List((0.0,Vector(100.0)), (2.0,Vector(391.0)), (4.0,Vector(50.0)), (6.0
                                                  //| ,Vector(13.0)), (8.0,Vector(28.0)), (10.0,Vector(173.0)), (12.0,Vector(394.
                                                  //| 0)), (14.0,Vector(22.0)), (16.0,Vector(12.0)), (18.0,Vector(57.0)), (20.0,V
                                                  //| ector(280.0)), (22.0,Vector(179.0)), (24.0,Vector(9.0)), (26.0,Vector(8.0))
                                                  //| , (28.0,Vector(75.0)), (30.0,Vector(364.0)))
  val mll = pfMLLik(1000, simPrior, 0.0, stepLV, obsLik, data)
                                                  //> mll  : bayeskit.sim.Parameter => Double = <function1>
  val mllSample = mll(Vector(1.0, 0.005, 0.6))    //> mllSample  : Double = -69.8812752856767

  val pmll = pfMLLikPar(1000, simPrior, 0.0, stepLV, obsLik, data)
                                                  //> pmll  : bayeskit.sim.Parameter => Double = <function1>
  val pmllSample = mll(Vector(1.0, 0.005, 0.6))   //> pmllSample  : Double = -70.1485876011123

}