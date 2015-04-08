package bayeskit

object pfilter {

  import scala.annotation.tailrec
  import sim._
  import scala.collection.parallel.immutable.{ ParSeq, ParVector }
  import breeze.linalg.DenseVector
  import breeze.stats.distributions.Multinomial
  import spire.math._
  import spire.implicits._

  def diff[T: Fractional](l: Iterable[T]): Iterable[T] = {
    (l.tail zip l) map { x => x._1 - x._2 }
  }

  def sample(n: Int, prob: DenseVector[Double]): Vector[Int] = {
    Multinomial(prob).sample(n).toVector
  }

  def mean[T: Numeric](it: Iterable[T]): Double = {
    it.map(_.toDouble).sum / it.size
  }

  def pfMLLik[S <: State, P <: Parameter, O <: Observation](
    n: Int,
    simx0: (Int, Time, P) => Vector[S],
    t0: Time,
    stepFun: (S, Time, Time, P) => S,
    dataLik: (S, O, P) => LogLik,
    data: TS[O]): (P => LogLik) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: P) => {
      val x0 = simx0(n, t0, th)
      @tailrec def pf(ll: LogLik, x: Vector[S], t: Time, deltas: Iterable[Time], obs: List[O]): LogLik =
        obs match {
          case Nil => ll
          case head :: tail => {
            val xp = if (deltas.head == 0) x else (x map { stepFun(_, t, deltas.head, th) })
            val lw = xp map { dataLik(_, head, th) }
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

  def pfProp[S <: State, P <: Parameter, O <: Observation](
    n: Int,
    simx0: (Int, Time, P) => Vector[S],
    t0: Time,
    stepFun: (S, Time, Time, P) => S,
    dataLik: (S, O, P) => LogLik,
    data: TS[O]): (P => (LogLik, List[S])) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: P) => {
      val x0 = simx0(n, t0, th)
      @tailrec def pf(ll: LogLik, x: Vector[List[S]], t: Time, deltas: Iterable[Time], obs: List[O]): (LogLik, List[S]) =
        obs match {
          case Nil => (ll, x(0).reverse)
          case head :: tail => {
            val xp = if (deltas.head == 0) x else (x map { l => stepFun(l.head, t, deltas.head, th) :: l })
            val lw = xp map { l => dataLik(l.head, head, th) }
            val max = lw.max
            val w = lw map { x => exp(x - max) }
            val rows = sample(n, DenseVector(w.toArray))
            val xpp = rows map { xp(_) }
            pf(ll + max + log(mean(w)), xpp, t + deltas.head, deltas.tail, tail)
          }
        }
      pf(0, x0 map { _ :: Nil }, t0, deltas, obs)
    }
  }

  def mean[A](it: ParSeq[A])(implicit n: Numeric[A]): Double = {
    it.map(n.toDouble).sum / it.size
  }

  def pfMLLikPar[S <: State, P <: Parameter, O <: Observation](
    n: Int,
    simx0: (Int, Time, P) => Vector[S],
    t0: Time,
    stepFun: (S, Time, Time, P) => S,
    dataLik: (S, O, P) => LogLik,
    data: TS[O]): (P => LogLik) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: P) => {
      val x0 = simx0(n, t0, th).par
      @tailrec def pf(ll: LogLik, x: ParVector[S], t: Time, deltas: Iterable[Time], obs: List[O]): LogLik =
        obs match {
          case Nil => ll
          case head :: tail => {
            val xp = if (deltas.head == 0) x else (x map { stepFun(_, t, deltas.head, th) })
            val lw = xp map { dataLik(_, head, th) }
            val max = lw.max
            //println(max)
            val w = lw map { x => exp(x - max) }
            val rows = sample(n, DenseVector(w.toArray)).par
            val xpp = rows map { xp(_) }
            pf(ll + max + log(mean(w)), xpp, t + deltas.head, deltas.tail, tail)
          }
        }
      pf(0, x0, t0, deltas, obs)
    }
  }

  def pfPropPar[S <: State, P <: Parameter, O <: Observation](
    n: Int,
    simx0: (Int, Time, P) => Vector[S],
    t0: Time,
    stepFun: (S, Time, Time, P) => S,
    dataLik: (S, O, P) => LogLik,
    data: TS[O]): (P => (LogLik, List[S])) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: P) => {
      val x0 = simx0(n, t0, th).par
      @tailrec def pf(ll: LogLik, x: ParVector[List[S]], t: Time, deltas: Iterable[Time], obs: List[O]): (LogLik, List[S]) =
        obs match {
          case Nil => (ll, x(0).reverse)
          case head :: tail => {
            val xp = if (deltas.head == 0) x else (x map { l => stepFun(l.head, t, deltas.head, th) :: l })
            val lw = xp map { l => dataLik(l.head, head, th) }
            val max = lw.max
            val w = lw map { x => exp(x - max) }
            val rows = sample(n, DenseVector(w.toArray)).par
            val xpp = rows map { xp(_) }
            pf(ll + max + log(mean(w)), xpp, t + deltas.head, deltas.tail, tail)
          }
        }
      pf(0, x0 map { _ :: Nil }, t0, deltas, obs)
    }
  }

}

