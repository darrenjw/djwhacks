/*
mh-ppl.scala

PPL encoding a log-likelihood function that can be used with a MH algorithm

*/

import breeze.stats.{distributions => bdist}
import breeze.linalg.DenseVector

trait Prob[T] {
  val ll: Vector[Double] => Double
  val pd: Int // dimension of the parameter vector
  def flatMap[U](f: T => Prob[U]): Prob[U] =
    LogLik(v => ll(v.take(pd)) + f(v.tail).ll(v.head))
  def map[U](f: T => U) = flatMap(t => Dirac(f(t)))
}

case class LogLik[T](ll: Vector[Double] => Double) extends Prob[T]

case class Dirac[T](x: T) extends Prob[T] {
  val ll = (obs: T) => if (obs == x)
    Double.PositiveInfinity
  else
    Double.NegativeInfinity
}

case class Normal(mu: Double, v: Double) extends Prob[Double] {
  val ll = (obs: Double) => bdist.Gaussian(mu, math.sqrt(v)).logPdf(obs)
}
 
case class Gamma(a: Double, b: Double) extends Prob[Double] {
  val ll = (obs: Double) => bdist.Gamma(a, 1.0/b).logPdf(obs)
}
 
case class Poisson(mu: Double) extends Prob[Int] {
  val ll = (obs: Int) => bdist.Poisson(mu).logProbabilityOf(obs)
}

object PplApp {

  def main(args: Array[String]): Unit = {
    println(breeze.stats.distributions.Poisson(10).sample(5))
    val pp = for {
      mu <- Normal(0.0,1.0)
      v <- Gamma(1.0,1.0)
      x <- Normal(mu,v)
    } yield x
  }

}


// eof

