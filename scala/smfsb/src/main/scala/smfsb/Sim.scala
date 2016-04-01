/*
Sim.scala

 */

package smfsb

import Types._
import annotation.tailrec

object Sim {

  def simTs[S: State](
    x0: S,
    t0: Time,
    tt: Time,
    dt: Time,
    stepFun: (S, Time, Time) => S
  ): Ts[S] = {
    @tailrec def simTsList(
      list: Ts[S],
      tt: Time,
      dt: Time,
      stepFun: (S, Time, Time) => S
    ): Ts[S] = {
      val (t0, x0) = list.head
      if (t0 >= tt) list else {
        val t1 = t0 + dt
        val x1 = stepFun(x0, t0, dt)
        simTsList((t1, x1) :: list, tt, dt, stepFun)
      }
    }
    simTsList(List((t0, x0)), tt, dt, stepFun).reverse
  }

  def plotTs[S: State](ts: Ts[S]): Unit = {
    import breeze.plot._
    import breeze.linalg._
    val times = DenseVector((ts map (_._1)).toArray)
    val idx = 0 until ts(0)._2.toDvd.length
    val states = ts map (_._2)
    val f = Figure()
    val p = f.subplot(0)
    idx.foreach(i => p += plot(times, DenseVector((states map (_.toDvd.apply(i))).toArray)))
    p.xlabel = "Time"
    p.ylabel = "Species count"
    f.saveas("TsPlot.png")
  }

  def toCsv[S: State](ts: Ts[S]): String = {
    val ls = ts map { t => t._1.toString + "," + t._2.toCsv + "\n" }
    ls.foldLeft("")(_ + _)
  }

  // TODO: Add a simSample function

}

/* eof */

