/*
min-ppl.scala

A minimal probabilistic programming language

*/

object MinPpl {

  // *************************************************************************
  // Code for the language starts here

  import breeze.stats.{distributions => bdist}
  import breeze.linalg.DenseVector

  implicit val numParticles = 2500

  case class Particle[T](v: T, lw: Double) { // value and log-weight
    def map[S](f: T => S): Particle[S] = Particle(f(v), lw)
    def flatMap[S](f: T => Particle[S]): Particle[S] = { // TODO: Don't need flatMap??
      val ps = f(v)
      Particle(ps.v, lw + ps.lw)
    }
  }

  trait Prob[T] {
    val particles: Vector[Particle[T]]
    def map[S](f: T => S): Prob[S] = Empirical(particles map (_ map f))
    def flatMap[S](f: T => Prob[S]): Prob[S] = {
      Empirical((particles map (p => {
        f(p.v).particles.map(psi => Particle(psi.v, p.lw + psi.lw))
      })).flatten).resample
    }
    def resample(implicit N: Int): Prob[T] = {
      val lw = particles map (_.lw)
      val mx = lw reduce (math.max(_,_))
      val rw = (lw map (_ - mx)) map (math.exp(_))
      val ind = bdist.Multinomial(DenseVector(rw.toArray)).sample(N)
      val newParticles = ind map (i => particles(i))
      Empirical(newParticles.toVector map (pi => Particle(pi.v, 0.0)))
    }
    def cond(ll: T => Double)(implicit N: Int): Prob[T] =
      Empirical(particles map (p => Particle(p.v, p.lw + ll(p.v))))
    def empirical: Vector[T] = resample.particles.map(_.v)
  }

  case class Empirical[T](particles: Vector[Particle[T]]) extends Prob[T]

  def unweighted[T](ts: Vector[T]): Prob[T] = Empirical(ts map (Particle(_, 0.0)))

  trait Dist[T] extends Prob[T] {
    def ll(obs: T): Double
    def ll(obs: Seq[T]): Double = obs map (ll) reduce (_+_)
  }
  // TODO: possible to add a "fit" method, a la Rainier?

  case class Normal(mu: Double, v: Double)(implicit N: Int) extends Dist[Double] {
    lazy val particles = unweighted(bdist.Gaussian(mu, math.sqrt(v)).sample(N).toVector).particles
    def ll(obs: Double) = bdist.Gaussian(mu, math.sqrt(v)).logPdf(obs)
  }

  case class Gamma(a: Double, b: Double)(implicit N: Int) extends Prob[Double] {
    lazy val particles = unweighted(bdist.Gamma(a, 1.0/b).sample(N).toVector).particles
    def ll(obs: Double) = bdist.Gamma(a, 1.0/b).logPdf(obs)
  }

  case class Poisson(mu: Double)(implicit N: Int) extends Prob[Int] {
    lazy val particles = unweighted(bdist.Poisson(mu).sample(N).toVector).particles
    def ll(obs: Int) = bdist.Poisson(mu).logProbabilityOf(obs)
  }

  // Code for the language ends here
  // *************************************************************************


  // Now a few simple examples

  // Linear Gaussian
  def example1 = {
    val xy = for {
      x <- Normal(5,4)
      y <- Normal(x,1)
    } yield (x,y)
    val y = xy.map(_._2)
    val yGz = y.cond(yi => Normal(yi, 9).ll(8.0)).empirical
    println(breeze.stats.meanAndVariance(y.empirical))
    println(breeze.stats.meanAndVariance(yGz))
    val xyGz = xy.cond{case (x,y) => Normal(y,9).ll(8.0)}.empirical
    println(breeze.stats.meanAndVariance(xyGz.map(_._1))) // x
    println(breeze.stats.meanAndVariance(xyGz.map(_._2))) // y
    // Now cond inside for expression...
    val xyz = for {
      x <- Normal(5,4)
      y <- Normal(x,1).cond(y => Normal(y,9).ll(8.0))
    } yield (x,y)
    val xyze = xyz.empirical
    println(breeze.stats.meanAndVariance(xyze.map(_._1))) // x
    println(breeze.stats.meanAndVariance(xyze.map(_._2))) // y
  }
  // TODO: do analytic checks

  // Normal random sample
  def example2 = {
    val prior = for {
      mu <- Normal(0,100)
      v <- Gamma(1,0.01)
    } yield (mu,v)
    val mod = prior.cond{case (mu,v) => Normal(mu,v).ll(List(8.0,9,7,7,8,10))}.empirical
    println(breeze.stats.meanAndVariance(mod map (_._1)))
    println(breeze.stats.meanAndVariance(mod map (_._2)))
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

