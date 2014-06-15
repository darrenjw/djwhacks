package bayeskit


object pfilter {

  import scala.annotation.tailrec
  import sim._
  import scala.collection.parallel.immutable.{ParSeq,ParVector}
  import breeze.linalg.DenseVector
  import breeze.stats.distributions.Multinomial
  
  // R-like "diff" function
  def diff(l: List[Double]): List[Double] = {
    (l.tail zip l) map { x => x._1 - x._2 }
  }
  
  def sample(n: Int, prob: DenseVector[Double]): Vector[Int] = {
    Multinomial(prob).sample(n).toVector
  }
  
  def mean[A](it:Iterable[A])(implicit n:Numeric[A]): Double = {
    it.map(n.toDouble).sum / it.size
  }
  
  def pfMLLik(
    n: Int,
    simx0: (Int, Time, Parameter) => Vector[State],
    t0: Double,
    stepFun: (State, Time, Time, Parameter) => State,
    dataLik: (State, Observation, Parameter) => Double,
    data: ObservationTS): (Parameter => Option[Double]) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: Parameter) => {
      val x0 = simx0(n, t0, th)
      @tailrec def pf(ll: Double, x: Vector[State], t: Time, deltas: List[Time], obs: List[Observation]): Option[Double] =
        obs match {
          case Nil => Some(ll)
          case head :: tail => {
            val xp = if (deltas.head==0) x else (x map { stepFun(_, t, deltas.head, th) })
            val w = xp map { dataLik(_, head, th) }
            if (w.sum < 1.0e-90) {
              System.err.print("\nParticle filter bombed with parameter " + th + "\n")
              None
            }
            val rows = sample(n, DenseVector(w.toArray))
            val xpp = rows map { xp(_) }
            pf(ll + math.log(mean(w)), xpp, t + deltas.head, deltas.tail, tail)
          }
        }
      pf(0, x0, t0, deltas, obs)
    }
  }

  def pfProp(
    n: Int,
    simx0: (Int, Time, Parameter) => Vector[State],
    t0: Double,
    stepFun: (State, Time, Time, Parameter) => State,
    dataLik: (State, Observation, Parameter) => Double,
    data: ObservationTS): (Parameter => Option[(Double, List[State])]) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: Parameter) => {
      val x0 = simx0(n, t0, th)
      @tailrec def pf(ll: Double, x: Vector[List[State]], t: Time, deltas: List[Time], obs: List[Observation]): Option[(Double, List[State])] =
        obs match {
          case Nil => Some((ll, x(0).reverse))
          case head :: tail => {
            val xp = if (deltas.head==0) x else (x map { l => stepFun(l.head, t, deltas.head, th) :: l })
            val w = xp map { l => dataLik(l.head, head, th) }
            if (w.sum < 1.0e-90) {
              System.err.print("\nParticle filter bombed with parameter " + th + "\n")
              None
            } else {
              val rows = sample(n, DenseVector(w.toArray))
              val xpp = rows map { xp(_) }
              pf(ll + math.log(mean(w)), xpp, t + deltas.head, deltas.tail, tail)
            }
          }
        }
      pf(0, x0 map { _ :: Nil }, t0, deltas, obs)
    }
  }

  def mean[A](it:ParSeq[A])(implicit n:Numeric[A]): Double = {
    it.map(n.toDouble).sum / it.size
  }


  def pfMLLikPar(
    n: Int,
    simx0: (Int, Time, Parameter) => Vector[State],
    t0: Double,
    stepFun: (State, Time, Time, Parameter) => State,
    dataLik: (State, Observation, Parameter) => Double,
    data: ObservationTS): (Parameter => Option[Double]) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: Parameter) => {
      val x0 = simx0(n, t0, th).par
      @tailrec def pf(ll: Double, x: ParVector[State], t: Time, deltas: List[Time], obs: List[Observation]): Option[Double] =
        obs match {
          case Nil => Some(ll)
          case head :: tail => {
            val xp = if (deltas.head==0) x else  (x map { stepFun(_, t, deltas.head, th) })
            val w = xp map { dataLik(_, head, th) }
            if (w.sum < 1.0e-90) {
              System.err.print("\nParticle filter bombed with parameter " + th + "\n")
              None
            }
            val rows = sample(n, DenseVector(w.toArray)).par
            val xpp = rows map { xp(_) }
            pf(ll + math.log(mean(w)), xpp, t + deltas.head, deltas.tail, tail)
          }
        }
      pf(0, x0, t0, deltas, obs)
    }
  }

  def pfPropPar(
    n: Int,
    simx0: (Int, Time, Parameter) => Vector[State],
    t0: Double,
    stepFun: (State, Time, Time, Parameter) => State,
    dataLik: (State, Observation, Parameter) => Double,
    data: ObservationTS): (Parameter => Option[(Double, List[State])]) = {
    val (times, obs) = data.unzip
    val deltas = diff(t0 :: times)
    (th: Parameter) => {
      val x0 = simx0(n, t0, th).par
      @tailrec def pf(ll: Double, x: ParVector[List[State]], t: Time, deltas: List[Time], obs: List[Observation]): Option[(Double, List[State])] =
        obs match {
          case Nil => Some((ll, x(0).reverse))
          case head :: tail => {
            val xp = if (deltas.head==0) x else (x map { l => stepFun(l.head, t, deltas.head, th) :: l })
            val w = xp map { l => dataLik(l.head, head, th) }
            if (w.sum < 1.0e-90) {
              System.err.print("\nParticle filter bombed with parameter " + th + "\n")
              None
            } else {
              val rows = sample(n, DenseVector(w.toArray)).par
              val xpp = rows map { xp(_) }
              pf(ll + math.log(mean(w)), xpp, t + deltas.head, deltas.tail, tail)
            }
          }
        }
      pf(0, x0 map { _ :: Nil }, t0, deltas, obs)
    }
  }

}