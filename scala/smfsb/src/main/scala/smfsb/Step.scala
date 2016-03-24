/*
Step.scala

Simulation algorithms, including Gillespie

 */

package smfsb

import annotation.tailrec
import breeze.linalg._
import breeze.stats.distributions.{Exponential, Multinomial}
import Types._

object Step {

  def gillespie[P <: Parameter](n: Spn[P]): (P) => (State, Time, Time) => State = {
    val S = (n.post - n.pre).t
    val v = S.cols
    (th: P) => (x, t0, dt) => {
      @tailrec def go(x: State, t0: Time, dt: Time): State = {
        if (dt <= 0.0) x else {
          val h = n.h(th)(x, t0)
          val h0 = sum(h)
          val t = if (h0 < 1e-50) 1e99 else new Exponential(h0).draw
          if (t > dt) x else {
            val i = Multinomial(h).sample
            go(x + S(::, i), t0 + t, dt - t)
          }
        }
      }
      go(x, t0, dt)
    }
  }

  // TODO: add a simple Poisson time stepping (tau leap) algorithm, too...

}

/* eof */

