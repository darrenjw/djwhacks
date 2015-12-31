/* superadder.scala

Illustration of Monoid in scalaz

 */

import scalaz.Monoid
import scalaz.syntax.monoid._

object SuperAdder {
  def addItems[A: Monoid](items: List[A]): A =
    items.foldLeft(mzero[A])(_ |+| _)

  def examples(): Unit = {
    import scalaz.std.option._
    import scalaz.std.anyVal._
    println(addItems(List(1, 2, 3)))
    println(addItems(List(Option(1), Option(2), Option(3))))
  }

  def main(args: Array[String]): Unit = {
    examples()
  }

}

// Given a subtyping relationship between A and B (e.g. A <: B)
// what is the relationship between F[A] and F[B]
// Possible answers:
// 1. No relationship. Invariance.
// 2. F[A] <: F[B]. Covariance.
// 3. F[A] >: F[B]. Contravariance

// Multiplication[Int]
// Multiplication[Double]
