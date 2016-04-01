/*
Mll.scala
Code for computing estimates of marginal likelihood (typically using a bootstrap PF)

 */

package smfsb

import annotation.tailrec

object Mll {

  import breeze.linalg.DenseVector
  import breeze.stats.distributions.Multinomial
  import spire.math._
  import spire.implicits._
  import Types._

  def diff[T: Fractional](l: Iterable[T]): Iterable[T] = {
    (l.tail zip l) map { x => x._1 - x._2 }
  }

  def sample(n: Int, prob: DenseVector[Double]): Vector[Int] = {
    Multinomial(prob).sample(n).toVector
  }

  def mean[T: Numeric](it: Iterable[T]): Double = {
    it.map(_.toDouble).sum / it.size
  }

  def pfMll[S: State, P: Parameter, O: Observation](
    n: Int,
    simx0: (P) => (Int, Time) => Vector[S],
    t0: Time,
    stepFun: (P) => (S, Time, Time) => S,
    dataLik: (P) => (S, O) => LogLik,
    data: Ts[O]
  ): (P => LogLik) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: P) => {
      val x0 = simx0(th)(n, t0)
      @tailrec def pf(ll: LogLik, x: Vector[S], t: Time, deltas: Iterable[Time], obs: List[O]): LogLik =
        obs match {
          case Nil => ll
          case head :: tail => {
            val xp = if (deltas.head == 0) x else (x map { stepFun(th)(_, t, deltas.head) })
            val lw = xp map { dataLik(th)(_, head) }
            val max = lw.max
            val w = lw map { x => exp(x - max) }
            val rows = sample(n, DenseVector(w.toArray))
            val xpp = rows map { xp(_) }
            pf(ll + max + log(mean(w)), xpp, t + deltas.head, deltas.tail, tail)
          }
        }
      pf(0, x0, t0, deltas, obs)
    }
  }


  // Now a parallel implementation, for slow forward models

  import scala.collection.parallel.immutable.{ParSeq, ParVector}

  def mean[A](it: ParSeq[A])(implicit n: Numeric[A]): Double = {
    it.map(n.toDouble).sum / it.size
  }

  def pfMllP[S: State, P: Parameter, O: Observation](
    n: Int,
    simx0: (P) => (Int, Time) => Vector[S],
    t0: Time,
    stepFun: (P) => (S, Time, Time) => S,
    dataLik: (P) => (S, O) => LogLik,
    data: Ts[O]
  ): (P => LogLik) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: P) => {
      val x0 = simx0(th)(n, t0).par
      @tailrec def pf(ll: LogLik, x: ParVector[S], t: Time, deltas: Iterable[Time], obs: List[O]): LogLik =
        obs match {
          case Nil => ll
          case head :: tail => {
            val xp = if (deltas.head == 0) x else (x map { stepFun(th)(_, t, deltas.head) })
            val lw = xp map { dataLik(th)(_, head) }
            val max = lw.max
            val w = lw map { x => exp(x - max) }
            val rows = sample(n, DenseVector(w.toArray)).par
            val xpp = rows map { xp(_) }
            pf(ll + max + log(mean(w)), xpp, t + deltas.head, deltas.tail, tail)
          }
        }
      pf(0, x0, t0, deltas, obs)
    }
  }

}

/* eof */

