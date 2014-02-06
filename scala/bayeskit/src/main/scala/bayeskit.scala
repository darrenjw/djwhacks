package bayeskit

object bayeskit {

  import breeze.stats.distributions._

  class State(val prey: Int, val predator: Int) {
    override def toString="("+prey.toString+","+predator.toString+")"
  }

  class Parameter(val c0: Double, val c1: Double, val c2: Double)

  @annotation.tailrec
  def stepLV(x: State, t0: Double, dt: Double, th: Parameter): State = {
    if (dt <= 0.0) x else {
      val h = (th.c0 * x.prey, th.c1 * x.predator * x.prey, th.c2 * x.predator)
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

  def simTs(x0: State,
    t0: Double,
    tt: Double,
    dt: Double,
    stepFun: (State, Double, Double, Parameter) => State,
    th: Parameter): List[(Double, State)] = {
    def simTsList(list: List[(Double, State)],
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

  def main(args: Array[String]): Unit = {
    println("hello")
    val state = stepLV(new State(100, 50), 0, 10, new Parameter(1.0, 0.005, 0.6))
    println(state.toString)
    val ts = simTs(new State(100, 50), 0, 100, 0.1, stepLV, new Parameter(1.0, 0.005, 0.6))
    println(ts)
    println("goodbye")
  }

}