

object Mean {

import breeze.stats.distributions._
import breeze.linalg._

  def mean[A](it:Iterable[A])(implicit n:Numeric[A]): Double = {
    it.map(n.toDouble).sum / it.size
  }

  import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution
  def sample(n: Int, prob: Array[Double]): Vector[Int] = {
    val inds = (0 to (prob.length - 1)).toArray
    val cat = new EnumeratedIntegerDistribution(inds, prob)
    (inds map { x => cat.sample }).toVector
  }

  def sample(n: Int, prob: DenseVector[Double]): IndexedSeq[Int] = {
    Multinomial(prob).sample(n)
  }

  def main(args: Array[String]): Unit = {
    println("hello")
    val v=Vector(1,2,3,4,5,6)
    println(mean(v))
    println(mean(Vector(1,2,3)))
    println(mean(List(2,3,4)))
    println(mean(Vector(1.0,1.2,1.6)))
    val p=Poisson(5.0).sample(100000)
    println(mean(p))
    println("goodbye")
  }

}
