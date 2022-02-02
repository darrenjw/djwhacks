/*

free-ppl.scala


 */

object FreePPL:

  enum Dist[A]:
    //case Observe(d: Dist[A], o: A) extends Dist[A]
    case Cond(ll: A => Double) extends Dist[A]
    case Normal(m: Double, s: Double) extends Dist[Double]
    case Poisson(l: Double) extends Dist[Int]
    case Gamma(a: Double, b: Double) extends Dist[Double]
  import Dist.*

  import cats.free.Free
  type DistF[A] = Free[Dist, A]

  // def observe[A](d: Dist[A], o: A) = Free.liftF[Dist, A](Observe(d, o))
  def cond[A](ll: A => Double) = Free.liftF[Dist, A](Cond(ll))
  def normal(m: Double, s: Double) = Free.liftF[Dist, Double](Normal(m, s))
  def poisson(l: Double) = Free.liftF[Dist, Int](Poisson(l))
  def gamma(a: Double, b: Double) = Free.liftF[Dist, Double](Gamma(a, b))

object FreeExample:

  import FreePPL.*

  val prior1 = for
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
    
  def addTimePoint(current: DistF[(Double, List[Double])],
    obs: Int): DistF[(Double, List[Double])] =
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
      //_ <- observe(poisson(math.exp(st)), obs)
    yield (w, states)
    updated

  val mod = data.foldLeft(prior)(addTimePoint(_,_))




// eof

