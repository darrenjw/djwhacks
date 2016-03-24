/*
Step.scala

Simulation algorithms, including Gillespie

 */

package smfsb

import annotation.tailrec
import breeze.linalg._
import breeze.stats.distributions.{Exponential, Multinomial, Poisson}
import Types._

object Step {

  def gillespie[P <: Parameter](n: Spn[P]): (P) => (State, Time, Time) => State = {
    val S = (n.post - n.pre).t
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

  def pts[P <: Parameter](n: Spn[P], dt: Double = 0.01): (P) => (State, Time, Time) => State = {
    val S = (n.post - n.pre).t
    val v = S.cols
    (th: P) => (x, t0, deltat) => {
      @tailrec def go(x: State, t0: Time, deltat: Time): State = {
        if (deltat <= 0.0) x else {
          val adt = if (dt > deltat) deltat else dt
          val h = n.h(th)(x, t0)
          val r = h map (hi => Poisson(hi * adt).sample)
          val nx = x + (S * r).toDenseVector
          val tnx = nx map (xi => max(xi,0))
          go(tnx, t0 + adt, deltat - adt)
        }
      }
      go(x, t0, deltat)
    }
  }

  // TODO: add a simple adaptive time-stepping tau-leap algorithm, too...

}

/* eof */

