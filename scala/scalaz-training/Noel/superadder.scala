import scalaz.Monoid
import scalaz.syntax.monoid._

object SuperAdder {
  def addItems[A : Monoid](items: List[A]): A =
    items.foldLeft(mzero[A])(_ |+| _)

  def examples = {
    import scalaz.std.option._
    import scalaz.std.anyVal._

    addItems(List(1, 2, 3))
    addItems(List(Option(1), Option(2), Option(3)))
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
