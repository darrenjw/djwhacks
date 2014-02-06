package bayeskit

object bayeskit {

  // import breeze.stats.distributions._

  class State(val prey: Int, val predator: Int)

  class Parameter(val c0: Double, val c1: Double, val c2: Double)

  @annotation.tailrec
  def stepLV(x: State, t0: Double, dt: Double, th: Parameter): State = {
    if (dt <= 0.0) x else {
      val h = (th.c0 * x.prey, th.c1 * x.predator * x.prey, th.c2 * x.prey)
      val h0 = h._1 + h._2 + h._3
      val t = if (h0 < 1e-10 || x.prey > 1e6) 1e99 else 1.0 / h0 // should be exponential with this mean
      if (t > dt) x else {
        val u = 0.2 // should be U(0,1)
        if (u < h._1 / h0) stepLV(new State(x.prey + 1, x.predator), t0 + t, dt - t, th)
        else {
          if (u < (h._1 + h._2) / h0) stepLV(new State(x.prey - 1, x.predator + 1), t0 + t, dt - t, th)
          else stepLV(new State(x.prey, x.predator - 1), t0 + t, dt - t, th)
        }
      }
    }
  }

  def main(args: Array[String]): Unit = {
    println("hello")
  }

}