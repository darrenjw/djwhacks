package bayeskit

object pfilter {

  import scala.annotation.tailrec
  import sim._

  // R-like "diff" function
  def diff(l: List[Double]): List[Double] = {
    (l.tail zip l) map { x => x._1 - x._2 }
  }

  import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution
  // R-like "sample" function, using Apache Commons Math
  // Note that the sampling here is WITH replacement, which is not the R default
  def sample(n: Int, prob: Vector[Double]): Vector[Int] = {
    val inds = (0 to (prob.length - 1)) toArray
    val cat = new EnumeratedIntegerDistribution(inds, prob.toArray)
    inds map { x => cat.sample } toVector
  }

  def mean(vec: Vector[Double]): Double = {
    vec.sum / vec.length
  }

  def pfMLLik(
    n: Int,
    simx0: (Int, Time, Parameter) => Vector[State],
    t0: Double,
    stepFun: (State, Time, Time, Parameter) => State,
    dataLik: (State, Observation, Parameter) => Double,
    data: ObservationTS): (Parameter => Double) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: Parameter) => {
      val x0 = simx0(n, t0, th)
      @tailrec def pf(ll: Double, x: Vector[State], t: Time, deltas: List[Time], obs: List[Observation]): Double = {
        if (obs.isEmpty) ll else { // pattern match here!!! 
          val xp = x map { stepFun(_, t, deltas.head, th) }
          val w = xp map { dataLik(_, obs.head, th) }
          val rows = sample(n, w)
          val xpp = rows map { xp(_) }
          pf(ll + math.log(mean(w)), xpp, t + deltas.head, deltas.tail, obs.tail)
        }
      }
      pf(0, x0, t0, deltas, obs)
    }
  }

}