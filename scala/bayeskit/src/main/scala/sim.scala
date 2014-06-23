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

  // First declare the basic types
  type Time = Double // Double or Float?
  type TS[O] = List[(Time,O)]

  // simulation utilities

  def simTs[S <: State, P <: Parameter](
    x0: S,
    t0: Time,
    tt: Time,
    dt: Time,
    stepFun: (S, Time, Time, P) => S,
    th: P): List[(Time, S)] = {
    @tailrec def simTsList(list: List[(Time, S)],
      tt: Time,
      dt: Time,
      stepFun: (S, Time, Time, P) => S,
      th: P): List[(Time, S)] = {
      val (t0, x0) = list.head
      if (t0 >= tt) list else {
        val t1 = t0 + dt
        val x1 = stepFun(x0, t0, dt, th)
        simTsList((t1, x1) :: list, tt, dt, stepFun, th)
      }
    }
    simTsList(List((t0, x0)), tt, dt, stepFun, th).reverse
  }

}