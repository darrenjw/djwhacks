/*
min-ppl.scala

A minimal probabilistic programming language

*/

object MinPpl {

  val N = 2000

  trait Prob[T] {
    val particles: Vector[T]
    def map[S](f: T => S): Prob[S] = Empirical(particles map f)
    def flatMap[S](f: T => Prob[S]): Prob[S] = {
      val fm = (particles.map(f).map((p: Prob[S]) => p.particles)).flatten
      val r = breeze.stats.distributions.Rand.randInt(fm.length)
      val ind = Vector.fill(N)(r.draw)
      Empirical(ind map (i => fm(i)))
    }
    def cond(ll: T => Double): Prob[T] = {
      val lw = particles map (ll(_))
      import breeze.linalg._
      val mx = lw reduce (math.max(_,_))
      val rw = (lw map (_ - mx)) map (math.exp(_))
      val ind = breeze.stats.distributions.
        Multinomial(DenseVector(rw.toArray)).sample(N)
      val newParticles = ind map (i => particles(i))
      Empirical(newParticles.toVector)
    }
    def cond[S](obs: S, ll: (S, T) => Double): Prob[T] = cond(ll(obs, _))
  }

  case class Empirical[T](particles: Vector[T]) extends Prob[T]

  trait Dist[T] extends Prob[T] {
    def ll(obs: T): Double
    def ll(obs: Seq[T]): Double = obs.map(ll).reduce(_+_)
  }

  case class Normal(mu: Double, v: Double) extends Dist[Double] {
    val particles = breeze.stats.distributions.
      Gaussian(mu, math.sqrt(v)).sample(N).toVector
    def ll(obs: Double) = breeze.stats.distributions.
      Gaussian(mu, math.sqrt(v)).logPdf(obs)
  }

  case class Gamma(a: Double, b: Double) extends Prob[Double] {
    val particles = breeze.stats.distributions.
      Gamma(a, 1.0/b).sample(N).toVector
    def ll(obs: Double) = breeze.stats.distributions.
      Gamma(a, 1.0/b).logPdf(obs)
  }

  def main(args: Array[String]): Unit = {
    println("Hi")
    val xy = for {
      x <- Normal(0,9)
      y <- Normal(x,1)
    } yield (x,y)
    val y = xy map (_._2)
    val yGz = y.cond(yi => Normal(yi, 1).ll(List(5.0,4,4,3,4,5,6)))
    println(breeze.stats.meanAndVariance(y.particles))
    println(breeze.stats.meanAndVariance(yGz.particles))
    println("Bye")
  }


}


// eof

