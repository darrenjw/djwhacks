/*
min-ppl.scala

A minimal probabilistic programming language

*/

object MinPpl {

  // code for the language starts here

  import breeze.stats.{distributions => bdist}
  import breeze.linalg.DenseVector

  implicit val numParticles = 2000

  trait Prob[T] {
    val particles: Vector[T]
    def map[S](f: T => S): Prob[S] = Empirical(particles map f)
    def flatMap[S](f: T => Prob[S])(implicit N: Int): Prob[S] = {
      val fm = (particles.map(f).map((p: Prob[S]) => p.particles)).flatten
      val r = bdist.Rand.randInt(fm.length)
      val ind = Vector.fill(N)(r.draw)
      Empirical(ind map (i => fm(i)))
    }
    def cond(ll: T => Double)(implicit N: Int): Prob[T] = {
      val lw = particles map (ll(_))
      val mx = lw reduce (math.max(_,_))
      val rw = (lw map (_ - mx)) map (math.exp(_))
      val ind = bdist.Multinomial(DenseVector(rw.toArray)).sample(N)
      val newParticles = ind map (i => particles(i))
      Empirical(newParticles.toVector)
    }
  }

  case class Empirical[T](particles: Vector[T]) extends Prob[T]

  trait Dist[T] extends Prob[T] {
    def ll(obs: T): Double
    def ll(obs: Seq[T]): Double = obs map (ll) reduce (_+_)
  }

  case class Normal(mu: Double, v: Double)(implicit N: Int) extends Dist[Double] {
    lazy val particles = bdist.Gaussian(mu, math.sqrt(v)).sample(N).toVector
    def ll(obs: Double) = bdist.Gaussian(mu, math.sqrt(v)).logPdf(obs)
  }

  case class Gamma(a: Double, b: Double)(implicit N: Int) extends Prob[Double] {
    lazy val particles = bdist.Gamma(a, 1.0/b).sample(N).toVector
    def ll(obs: Double) = bdist.Gamma(a, 1.0/b).logPdf(obs)
  }

  case class Poisson(mu: Double)(implicit N: Int) extends Prob[Int] {
    lazy val particles = bdist.Poisson(mu).sample(N).toVector
    def ll(obs: Int) = bdist.Poisson(mu).logProbabilityOf(obs)
  }

  // code for the language ends here

  // now a few simple examples

  def example1 = {
    val xy = for {
      x <- Normal(0,9)
      y <- Normal(x,1)
    } yield (x,y)
    val y = xy map (_._2)
    val yGz = y.cond(yi => Normal(yi, 1).ll(List(5.0,4,4,3,4,5,6)))
    println(breeze.stats.meanAndVariance(y.particles))
    println(breeze.stats.meanAndVariance(yGz.particles))
  }




  // main entry point

  def main(args: Array[String]): Unit = {
    println("Hi")
    example1
    println("Bye")
  }


}


// eof

