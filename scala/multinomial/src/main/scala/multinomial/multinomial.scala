/*
multinomial.scala

Multinomial sampling for generic collections, with particle filtering in mind...

 */

import breeze.stats.distributions.Binomial

object Multinomial {

  // will require scanLeft and drop
  // does not require weights to be normalised
  def mnResample(N: Int, w: Vector[Double]): Vector[Int] = {
    val sw = w.reduce(_ + _)
    w.scanLeft((N, sw))((p, w) => (if (p._1 > 0) p._1 - Binomial(p._1, w / p._2).draw else 0, p._2 - w)).
      drop(1).
      scanLeft((N, 0))((a, b) => (b._1, a._1 - b._1)).
      drop(1).
      map(_._2)
  }

  def main(args: Array[String]): Unit = {
    println("hi")
    val w = Vector(0.5, 0.2, 0.2, 0.1)
    val c = mnResample(10, w)
    println(c)
    println("bye")
  }

}

// eof

