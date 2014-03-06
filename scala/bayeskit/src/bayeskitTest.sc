object bayeskitTest {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

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

  val state = stepLV(Vector(100, 50), 0, 10, Vector(1.0, 0.005, 0.6))
                                                  //> state  : bayeskit.sim.State = Vector(38, 23)

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
                                                  //> truth  : bayeskit.sim.StateTS = List((0.0,Vector(100, 50)), (2.0,Vector(346
                                                  //| , 159)), (4.0,Vector(56, 460)), (6.0,Vector(7, 176)), (8.0,Vector(2, 64)), 
                                                  //| (10.0,Vector(3, 24)), (12.0,Vector(14, 6)), (14.0,Vector(99, 2)), (16.0,Vec
                                                  //| tor(712, 33)), (18.0,Vector(6, 851)), (20.0,Vector(0, 245)), (22.0,Vector(0
                                                  //| , 87)), (24.0,Vector(0, 21)), (26.0,Vector(0, 5)), (28.0,Vector(0, 0)), (30
                                                  //| .0,Vector(0, 0)))
  val data = truth map { x => (x._1, Vector(x._2(0).toDouble)) }
                                                  //> data  : List[(bayeskit.sim.Time, scala.collection.immutable.Vector[Double])
                                                  //| ] = List((0.0,Vector(100.0)), (2.0,Vector(346.0)), (4.0,Vector(56.0)), (6.0
                                                  //| ,Vector(7.0)), (8.0,Vector(2.0)), (10.0,Vector(3.0)), (12.0,Vector(14.0)), 
                                                  //| (14.0,Vector(99.0)), (16.0,Vector(712.0)), (18.0,Vector(6.0)), (20.0,Vector
                                                  //| (0.0)), (22.0,Vector(0.0)), (24.0,Vector(0.0)), (26.0,Vector(0.0)), (28.0,V
                                                  //| ector(0.0)), (30.0,Vector(0.0)))
  val mll = pfMLLik(100, simPrior, 0.0, stepLV, obsLik, data)
                                                  //> mll  : bayeskit.sim.Parameter => Double = <function1>
  val mllSample = mll(Vector(1.0, 0.005, 0.6))    //> mllSample  : Double = -67.28810506848146

  val pmll = pfMLLikPar(100, simPrior, 0.0, stepLV, obsLik, data)
                                                  //> pmll  : bayeskit.sim.Parameter => Double = <function1>
  val pmllSample = pmll(Vector(1.0, 0.005, 0.6))  //> pmllSample  : Double = -66.88768979497242

  mll(Vector(1.0, 0.005, 0.6))                    //> res9: Double = -64.1019962028424
  mll(Vector(1.0, 0.005, 0.6))                    //> res10: Double = -64.2440812783
  mll(Vector(1.0, 0.005, 0.6))                    //> res11: Double = -66.75247119628489

  pmll(Vector(1.0, 0.005, 0.6))                   //> res12: Double = -65.20228013516781
  pmll(Vector(1.0, 0.005, 0.6))                   //> res13: Double = -71.45802206085168
  pmll(Vector(1.0, 0.005, 0.6))                   //> res14: Double = -65.51876458664515

  Vector(1,2,3).map{_*2}                          //> res15: scala.collection.immutable.Vector[Int] = Vector(2, 4, 6)
  val pmmhOutput=runPmmh(100,Vector(1.0, 0.005, 0.6),mll)
                                                  //> 100
                                                  //| Accept
                                                  //| 99
                                                  //| Accept
                                                  //| 98
                                                  //| Accept
                                                  //| 97
                                                  //| Reject
                                                  //| 96
                                                  //| Accept
                                                  //| 95
                                                  //| Accept
                                                  //| 94
                                                  //| Reject
                                                  //| 93
                                                  //| Reject
                                                  //| 92
                                                  //| Reject
                                                  //| 91
                                                  //| Accept
                                                  //| 90
                                                  //| Accept
                                                  //| 89
                                                  //| Accept
                                                  //| 88
                                                  //| Accept
                                                  //| 87
                                                  //| Accept
                                                  //| 86
                                                  //| Reject
                                                  //| 85
                                                  //| Accept
                                                  //| 84
                                                  //| Accept
                                                  //| 83
                                                  //| Reject
                                                  //| 82
                                                  //| org.apache.commons.math3.exception.MathArithmeticException: array sums to z
                                                  //| ero
                                                  //| 	at org.apache.commons.math3.util.MathArrays.normalizeArray(MathArrays.ja
                                                  //| va:1296)
                                                  //| 	at org.apache.commons.math3.distribution.EnumeratedDistribution.<init>(E
                                                  //| numeratedDistribution.java:126)
                                                  //| 	at org.apache.commons.math3.distribution.EnumeratedIntegerDistribution.<
                                                  //| init>(EnumeratedIntegerDistribution.java:100)
                                                  //| 	at org.apache.commons.math3.distribution.EnumeratedIntegerDistribution.<
                                                  //| init>(EnumeratedIntegerDistribution.java:68)
                                                  //| 	at bayeskit.pfilter$.sample(pfilter.scala:19)
                                                  //| 	at bayeskit.pfi
                                                  //| Output exceeds cutoff limit.
  
  
  
}