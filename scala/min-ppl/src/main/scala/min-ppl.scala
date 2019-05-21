/*
min-ppl.scala

A minimal probabilistic programming language

*/

object MinPpl {

  val N = 100

  trait Prob[T] {
    val particles: Vector[T]
    def map[S](f: T => S): Prob[S] = Empirical(particles map f)
    def flatMap[S](f: T => Prob[S]): Prob[S] = {
      val fm = (particles.map(f).map((p: Prob[S]) => p.particles)).flatten
      // TODO: thin
      Empirical(fm)
    }
    def cond[S](obs: S, ll: (S,T) => Double): Prob[T] = {
      val lw = particles map (ll(obs,_))
      import breeze.linalg._
      val mx = lw reduce (math.min(_,_))
      val rw = (lw map (_ - mx)) map (math.exp(_))
      val ind = breeze.stats.distributions.
        Multinomial(DenseVector(rw.toArray)).sample(N)
      val newParticles = ind map (i => particles(i))
      Empirical(newParticles.toVector)
    }
  }

  case class Empirical[T](particles: Vector[T]) extends Prob[T]

  case class Normal(mu: Double, v: Double) extends Prob[Double] {
    val particles = breeze.stats.distributions.
      Gaussian(mu,math.sqrt(v)).sample(N).toVector
  }

  case class Gamma(a: Double, b: Double) extends Prob[Double] {
    val particles = breeze.stats.distributions.
      Gamma(a,b).sample(N).toVector
  }

  def main(args: Array[String]): Unit = {
    println("Hi")

    println("Bye")
  }

}


// eof

