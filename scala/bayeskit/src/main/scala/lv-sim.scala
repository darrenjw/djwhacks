package bayeskit

object bayeskit {

  import breeze.stats.distributions._
  import scala.annotation.tailrec

  class State(val prey: Int, val predator: Int) {
    override def toString = "(" + prey.toString + "," + predator.toString + ")"
  }

  type Parameter = Vector[Double]

  type Observation = Vector[Double]

  // R-like "diff" function
  def diff(l: List[Double]): List[Double] = {
    (l.tail zip l) map { x => x._1 - x._2 }
  }

  // R-like "sample" function, using Commons Math
  // Note that the sampling here is WITH replacement, which is not the R default
  import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution
  def sample(n: Int, prob: Vector[Double]): Vector[Int] = {
    val inds = (0 to (prob.length - 1))
    val cat = new EnumeratedIntegerDistribution(inds.toArray, prob.toArray)
    inds.toVector map { x => cat.sample }
  }

  def mean(vec: Vector[Double]): Double = {
    vec.sum / vec.length
  }

  @tailrec def stepLV(x: State, t0: Double, dt: Double, th: Parameter): State = {
    if (dt <= 0.0) x else {
      val h = (th(0) * x.prey, th(1) * x.predator * x.prey, th(2) * x.predator)
      val h0 = h._1 + h._2 + h._3
      val t = if (h0 < 1e-10 || x.prey > 1e6) 1e99 else new Exponential(h0).draw
      if (t > dt) x else {
        val u = new Uniform(0, 1).draw // use simpler function!
        if (u < h._1 / h0) stepLV(new State(x.prey + 1, x.predator), t0 + t, dt - t, th)
        else {
          if (u < (h._1 + h._2) / h0) stepLV(new State(x.prey - 1, x.predator + 1), t0 + t, dt - t, th)
          else stepLV(new State(x.prey, x.predator - 1), t0 + t, dt - t, th)
        }
      }
    }
  }

  def simTs(
    x0: State,
    t0: Double,
    tt: Double,
    dt: Double,
    stepFun: (State, Double, Double, Parameter) => State,
    th: Parameter): List[(Double, State)] = {
    @tailrec def simTsList(list: List[(Double, State)],
      tt: Double,
      dt: Double,
      stepFun: (State, Double, Double, Parameter) => State,
      th: Parameter): List[(Double, State)] = {
      val (t0, x0) = list.head
      if (t0 >= tt) list else {
        val t1 = t0 + dt
        val x1 = stepFun(x0, t0, dt, th)
        simTsList((t1, x1) :: list, tt, dt, stepFun, th)
      }
    }
    simTsList(List((t0, x0)), tt, dt, stepFun, th).reverse
  }

  def pfMLLik(n: Int,
    simx0: (Int, Double, Parameter) => Vector[State],
    t0: Double,
    stepFun: (State, Double, Double, Parameter) => State,
    dataLik: (State, Observation, Parameter) => Double,
    data: List[(Double, Observation)]): (Parameter => Double) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: Parameter) => {
      val x0 = simx0(n, t0, th)
      @tailrec def pf(ll: Double, x: Vector[State], t: Double, deltas: List[Double], obs: List[Observation]): Double = {
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


  
  // Just the "main" function below...
  
  def main(args: Array[String]): Unit = {
    println("hello")
    val state = stepLV(new State(100, 50), 0, 10, Vector(1.0, 0.005, 0.6))
    println(state.toString)
    val ts = simTs(new State(100, 50), 0, 100, 0.1, stepLV, Vector(1.0, 0.005, 0.6))
    println(ts)
    println("goodbye")
  }

  
}