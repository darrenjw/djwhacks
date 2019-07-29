/*
min-ppl-unweighted.scala

A minimal probabilistic programming language

Unweighted version, which doesn't seem very promising

*/

object MinPplUW {

  // *************************************************************************
  // Code for the language starts here

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
  // TODO: possible to add a "fit" method, a la Rainier?

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

  // Code for the language ends here
  // *************************************************************************


  // Now a few simple examples

  // Linear Gaussian
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
  // TODO: do analytic checks
  // TODO: cond inside for expression?

  // Normal random sample
  def example2 = {
    val prior = for {
      mu <- Normal(0,100)
      v <- Gamma(1,0.01)
    } yield (mu,v)
    val mod = prior.cond{case (mu,v) => Normal(mu,v).ll(List(8.0,9,7,7,8,10))}
    println(breeze.stats.meanAndVariance(mod.particles map (_._1)))
    println(breeze.stats.meanAndVariance(mod.particles map (_._2)))
  }

  // TODO: Poisson DGLM
  def example3 = {

    val data = List(2,1,0,2,3,4,5,4,3,2,1)

    val prior = for {
      w <- Gamma(1, 0.01)
      state0 <- Normal(0.0, 100.0)
    } yield (w, List(state0))
    
    def addTimePoint(current: Prob[(Double, List[Double])],
      obs: Int): Prob[(Double, List[Double])] = for {
      tup <- current
        w = tup._1
        states = tup._2
        os = states.head
        ns <- Normal(os, w) // cond?
    } yield (w, ns :: states)

  }




  // TODO: Linear model





  // Main entry point

  def main(args: Array[String]): Unit = {
    println("Hi")
    example1
    println("Bye")
  }


}


// eof

