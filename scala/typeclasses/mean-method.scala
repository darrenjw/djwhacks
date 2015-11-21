/* generic mean method */

import spire.math._
import spire.implicits._

object MeanM {

  trait Mean[T] {
    def mean[T](it: Iterable[T]): Double
  }
  object Mean {
    def mean[T: Numeric](it: Iterable[T]): Double = {
      it.map(_.toDouble).sum / it.size
    }
  }
  implicit class MeanIC[T: Numeric](it: Iterable[T]) {
    def mean[T] = Mean.mean(it)
  }

  def main(args: Array[String]): Unit = {
    println(List(1, 2, 3).mean)
    println(Vector(1.0, 2.0, 2.0, 2.0).mean)
    import Mean.mean
    println(mean(List(1, 2)))
  }

}

/* eof */

