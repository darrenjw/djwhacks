package bayeskit

object sim {

  import scala.annotation.tailrec

  // First declare the basic types

  type Time = Double

  type State = Vector[Int]
  type StateTS = List[(Time, State)]

  type Parameter = Vector[Double]

  type Observation = Vector[Double]
  type ObservationTS = List[(Time, Observation)]

  // simulation utilities

  def simTs(
    x0: State,
    t0: Time,
    tt: Time,
    dt: Time,
    stepFun: (State, Time, Time, Parameter) => State,
    th: Parameter): StateTS = {
    @tailrec def simTsList(list: List[(Time, State)],
      tt: Time,
      dt: Time,
      stepFun: (State, Time, Time, Parameter) => State,
      th: Parameter): List[(Time, State)] = {
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