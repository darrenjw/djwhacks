/* generic mean function */

import spire.math._
import spire.implicits._

object Mean {

  def mean[T: Numeric](it: Iterable[T]): Double = {
    it.map(_.toDouble).sum / it.size
  }

  def main(args: Array[String]): Unit = {
    println(mean(List(1, 2, 3)))
    println(mean(Vector(1.0, 2.0, 2.0, 2.0)))
  }

}

/* eof */

