package bayeskit

object lvsim {

  import breeze.stats.distributions._
  import scala.annotation.tailrec

  import sim._

  @tailrec def stepLV(x: State, t0: Time, dt: Time, th: Parameter): State = {
    if (dt <= 0.0) x else {
      val h = (th(0) * x(0), th(1) * x(1) * x(0), th(2) * x(1))
      val h0 = h._1 + h._2 + h._3
      val t = if (h0 < 1e-10 || x(0) > 1e6) 1e99 else new Exponential(h0).draw
      if (t > dt) x else {
        val u = new Uniform(0, 1).draw // use simpler function!
        if (u < h._1 / h0) stepLV(Vector(x(0) + 1, x(1)), t0 + t, dt - t, th)
        else {
          if (u < (h._1 + h._2) / h0) stepLV(Vector(x(0) - 1, x(1) + 1), t0 + t, dt - t, th)
          else stepLV(Vector(x(0), x(1) - 1), t0 + t, dt - t, th)
        }
      }
    }
  }

}