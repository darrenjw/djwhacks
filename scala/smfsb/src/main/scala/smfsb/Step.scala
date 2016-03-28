/*
Step.scala

Simulation algorithms, including Gillespie

 */

package smfsb

import annotation.tailrec
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import Types._

object Step {

  def gillespie[P: Parameter](n: Spn[P, IntState]): (P) => (IntState, Time, Time) => IntState = {
    val Sto = (n.post - n.pre).t
    (th: P) => (x: IntState, t0, dt) => {
      @tailrec def go(x: IntState, t0: Time, dt: Time): IntState = {
        if (dt <= 0.0) x else {
          val h = n.h(th)(x, t0)
          val h0 = sum(h)
          val t = if (h0 < 1e-50) 1e99 else new Exponential(h0).draw
          if (t > dt) x else {
            val i = Multinomial(h).sample
            go(x + Sto(::, i), t0 + t, dt - t)
          }
        }
      }
      go(x, t0, dt)
    }
  }

  def pts[P: Parameter](n: Spn[P, IntState], dt: Double = 0.01): (P) => (IntState, Time, Time) => IntState = {
    val Sto = (n.post - n.pre).t
    val v = Sto.cols
    (th: P) => (x, t0, deltat) => {
      @tailrec def go(x: IntState, t0: Time, deltat: Time): IntState = {
        if (deltat <= 0.0) x else {
          val adt = if (dt > deltat) deltat else dt
          val h = n.h(th)(x, t0)
          val r = h map (hi => Poisson(hi * adt).sample)
          val nx = x + (Sto * r).toDenseVector
          val tnx = nx map (xi => max(xi, 0))
          go(tnx, t0 + adt, deltat - adt)
        }
      }
      go(x, t0, deltat)
    }
  }

  def cle[P: Parameter](n: Spn[P, DoubleState], dt: Double = 0.01): (P) => (DoubleState, Time, Time) => DoubleState = {
    val Sto = ((n.post - n.pre) map {_ * 1.0}).t
    val v = Sto.cols
    val sdt = Math.sqrt(dt)
    (th: P) => (x, t0, deltat) => {
      @tailrec def go(x: DoubleState, t0: Time, deltat: Time): DoubleState = {
        if (deltat <= 0.0) x else {
          val adt = if (dt > deltat) deltat else dt
          val sdt = Math.sqrt(adt)
          val h = n.h(th)(x, t0)
          val dw = DenseVector(Gaussian(0.0,sdt).sample(v).toArray)
          val dx = Sto * ((h * adt) + (sqrt(h) :* dw))
          val nx = x + dx.toDenseVector
          val tnx = abs(nx)
          go(tnx, t0 + adt, deltat - adt)
        }
      }
      go(x, t0, deltat)
    }
  }


  // TODO: general SDE E-M simulator

  // TODO: general ODE Euler method

  // TODO: method for converting cts SPNs to SDEs and ODEs

  // TODO: add a simple adaptive time-stepping tau-leap algorithm, too...

}

/* eof */

