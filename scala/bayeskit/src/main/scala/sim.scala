package bayeskit

trait State {
  def toString: String
}

trait Parameter {
  def toString: String
}

trait Observation

object sim {

  import scala.annotation.tailrec

  type Time = Double // Double or Float?
  
  type LogLik = Double
  
  type TS[O] = List[(Time,O)]

  def simTs[S <: State, P <: Parameter](
    x0: S,
    t0: Time,
    tt: Time,
    dt: Time,
    stepFun: (S, Time, Time, P) => S,
    th: P): TS[S] = {
    @tailrec def simTsList(list: TS[S],
      tt: Time,
      dt: Time,
      stepFun: (S, Time, Time, P) => S,
      th: P): TS[S] = {
      val (t0, x0) = list.head
      if (t0 >= tt) list else {
        val t1 = t0 + dt
        val x1 = stepFun(x0, t0, dt, th)
        simTsList((t1, x1) :: list, tt, dt, stepFun, th)
      }
    }
    simTsList(List((t0, x0)), tt, dt, stepFun, th).reverse
  }

  // TODO: Add a simSample function?
  
}