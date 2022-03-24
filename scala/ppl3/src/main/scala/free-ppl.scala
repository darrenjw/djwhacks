/*

free-ppl.scala


 */

import cats.*
import cats.syntax.*

object FreePPL:

  sealed trait Prob[A]
  case class Observe[A](d: Dist[A], o: A) extends Prob[A]
  //case Cond(ll: A => Double) extends Prob[A]

  sealed trait Dist[A] extends Prob[A]
  case class Normal(m: Double, v: Double) extends Dist[Double]
  case class Poisson(l: Double) extends Dist[Int]
  case class Gamma(a: Double, b: Double) extends Dist[Double]
 
  import cats.free.Free
  type ProbF[A] = Free[Prob, A]

  def observe[A](d: Dist[A], o: A) = Free.liftF[Prob, A](Observe(d, o))
  //def cond[A](ll: A => Double) = Free.liftF[Prob, A](Cond(ll))

  def normal(m: Double, s: Double) = Free.liftF[Prob, Double](Normal(m, s))
  def poisson(l: Double) = Free.liftF[Prob, Int](Poisson(l))
  def gamma(a: Double, b: Double) = Free.liftF[Prob, Double](Gamma(a, b))

object FreePPL3:

  import FreePPL.*

  val c2smc = new (Prob ~> PPL3.Prob):
    def dist[A](da: Dist[A]): PPL3.Dist[A] = da match
      case Normal(m, v) => PPL3.Normal(m, v)
      case Poisson(l) => PPL3.Poisson(l)
      case Gamma(a, b) => PPL3.Gamma(a, b)
    def apply[A](pa: Prob[A]): PPL3.Prob[A] = pa match
      case Observe(d: Dist[A], o: A) => dist(d).fitQ(o)
      case Normal(m, v) => dist(Normal(m, v))
      case Poisson(l) => dist(Poisson(l))
      case Gamma(a, b) => dist(Gamma(a,b))

object FreeExample:

  import FreePPL.*

  val prior1 = for
    x <- normal(0,1)
  yield (x)

  val prior2 = for
    x <- normal(0,1)
    y <- gamma(1,1)
  yield (x,y)

  val prior3 = for
    x <- normal(0,1)
    y <- gamma(1,1)
    z <- poisson(10)
  yield (x,y,z)

  // SSM example:

  val data = List(2,1,0,2,3,4,5,4,3,2,1)

  val prior = for
    w <- gamma(1, 1)
    state0 <- normal(0.0, 2.0)
  yield (w, List(state0))
    
  def addTimePoint(current: ProbF[(Double, List[Double])],
    obs: Int): ProbF[(Double, List[Double])] =
    println(s"Conditioning on observation: $obs")
    val predict = for
      tup <- current
      (w, states) = tup
      os = states.head
      ns <- normal(os, w)
    yield (w, ns :: states)
    val updated = for
      tup <- predict
      (w, states) = tup
      st = states.head
      _ <- observe(Poisson(math.exp(st)), obs)
    yield (w, states)
    updated

  val mod = data.foldLeft(prior)(addTimePoint(_,_))


  @main def runFree() =
    println("Starting...")
    import FreePPL3.c2smc
    import breeze.stats.{meanAndVariance => meanVar}
    println("PP1..")
    val pp1 = prior1.foldMap(c2smc)
    println(meanVar(pp1.empirical))
    println("PP2...")
    val pp2 = prior2.foldMap(c2smc)
    println(meanVar(pp2.empirical.map(_._1)))
    println(meanVar(pp2.empirical.map(_._2)))
    val pp3 = prior3.foldMap(c2smc) // takes 15 hours!
    println(meanVar(pp3.empirical.map(_._1)))
    println(meanVar(pp3.empirical.map(_._2)))
    println(meanVar(pp3.empirical.map(_._3.toDouble)))



// eof

