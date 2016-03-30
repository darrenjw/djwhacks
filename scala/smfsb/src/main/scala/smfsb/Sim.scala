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

  import breeze.linalg._
  def plotTsDouble(ts: Ts[DoubleState]): Unit = {
    import breeze.plot._
    import breeze.linalg._
    val times = DenseVector((ts map (_._1)).toArray)
    val idx = 0 until ts(0)._2.length
    val states = ts map (_._2)
    val f = Figure()
    val p = f.subplot(0)
    idx.foreach(i => p += plot(times, DenseVector((states map (_(i).toDouble)).toArray)))
    p.xlabel = "Time"
    p.ylabel = "Species count"
    f.saveas("TsPlot.png")
  }

  def plotTsInt(ts: Ts[IntState]): Unit = {
    val dts = ts map { t => (t._1, t._2.map { _ * 1.0 }) }
    plotTsDouble(dts)
  }

// TODO: Figure out how to dynamically dispatch the plot function...
// Easiest way is probably to introduce a "toDVD" method on State and Observation

/*

  def plotTs[S: State](ts: Ts[S]): Unit = {
    val s0 = (ts(0)._2)
    s0 match {
      case v: IntState => plotTsInt(ts)
      case v: DoubleState => plotTsDouble(ts)
    }
  }

 */

  // TODO: Add a simSample function

}

/* eof */

