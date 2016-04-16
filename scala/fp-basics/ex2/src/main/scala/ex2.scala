/*
ex2.scala

 */

object Ex2 {

  def findRootOpt(low: Double, high: Double)(f: Double => Double): Option[Double] = {
    if ((low < high) & (f(low) * f(high) < 0.0))
      Some(findRoot(low, high)(f)) else None
  }

  def findRoot(low: Double, high: Double)(f: Double => Double): Double = {
    findRoot(low, f(low), high)(f)
  }

  val tol = 1.0e-10

  @annotation.tailrec
  def findRoot(low: Double, fl: Double, high: Double)(f: Double => Double): Double = {
    val mid = 0.5 * (low + high)
    val fm = f(mid)
    if ((math.abs(fm) < tol) || (high - low < tol)) mid else {
      if (fl * fm > 0.0) findRoot(mid, fm, high)(f) else findRoot(low, fl, mid)(f)
    }
  }

}

/* eof */

