/*
mh-ppl.scala

PPL encoding a log-likelihood function that can be used with a MH algorithm

*/

import breeze.stats.{distributions => bdist}
import breeze.linalg.DenseVector


trait Prob[S, T] {
  val ll: S => Double
  val extract: S => T
  def flatMap[U](f: T => Dist[U]): Prob[(U, S), U] =
    LogLik((us: (U, S)) => ll(us._2) + f(extract(us._2)).ll(us._1),
      (us: (U, S)) => us._1)
}

case class LogLik[S, T](ll: S => Double, extract: S => T) extends Prob[S, T]

trait Dist[T] extends Prob[T, T] {
  val extract = identity
}

case class Normal(mu: Double, v: Double) extends Dist[Double] {
  val ll = (obs: Double) => bdist.Gaussian(mu, math.sqrt(v)).logPdf(obs)
}
 
case class Gamma(a: Double, b: Double) extends Dist[Double] {
  val ll = (obs: Double) => bdist.Gamma(a, 1.0/b).logPdf(obs)
}
 
case class Poisson(mu: Double) extends Dist[Int] {
  val ll = (obs: Int) => bdist.Poisson(mu).logProbabilityOf(obs)
}

object PplApp {

  def main(args: Array[String]): Unit = {
    println(breeze.stats.distributions.Poisson(10).sample(5))
  }

}


// eof

