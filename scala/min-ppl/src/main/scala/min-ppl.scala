/*
min-ppl.scala

A minimal probabilistic programming language

*/

object MinPpl {

  // *************************************************************************
  // Code for the language starts here

  import breeze.stats.{distributions => bdist}
  import breeze.linalg.DenseVector

  //implicit val numParticles = 100
  //implicit val numParticles = 300
  implicit val numParticles = 1000

  case class Particle[T](v: T, lw: Double) { // value and log-weight
    def map[S](f: T => S): Particle[S] = Particle(f(v), lw)
    // TODO: Don't need flatMap in min version
    // TODO: OR, re-write Prob.flatMap to use it...
    def flatMap[S](f: T => Particle[S]): Particle[S] = {
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
      //val np = particles.length ; println(s"$np $mx") // TODO: Debug code
      val rw = lw map (lwi => math.exp(lwi - mx))
      val law = mx + math.log(rw.sum/(rw.length))
      val ind = bdist.Multinomial(DenseVector(rw.toArray)).sample(N)
      val newParticles = ind map (i => particles(i))
      Empirical(newParticles.toVector map (pi => Particle(pi.v, law)))
    }
    // TODO: don't strictly need "cond" in a minimal PPL...
    def cond(ll: T => Double): Prob[T] =
      Empirical(particles map (p => Particle(p.v, p.lw + ll(p.v))))
    def empirical: Vector[T] = resample.particles.map(_.v)
  }

  case class Empirical[T](particles: Vector[Particle[T]]) extends Prob[T]

  def unweighted[T](ts: Vector[T]): Prob[T] = Empirical(ts map (Particle(_, 0.0)))

  trait Dist[T] extends Prob[T] {
    def ll(obs: T): Double
    def ll(obs: Seq[T]): Double = obs map (ll) reduce (_+_)
    def fit(obs: Seq[T]): Prob[T] = Empirical(particles map (p => Particle(p.v, p.lw + ll(obs))))
    def fitQ(obs: Seq[T]): Prob[T] = Empirical(Vector(Particle(obs.head, ll(obs))))
    def fit(obs: T): Prob[T] = fit(List(obs))
    def fitQ(obs: T): Prob[T] = fitQ(List(obs))
  }

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
    print("y: 5.000, 5.000 : ")
    println(breeze.stats.meanAndVariance(y.empirical))
    print("y: 6.071, 3.214 : ")
    println(breeze.stats.meanAndVariance(yGz))
    val xyGz = xy.cond{case (x,y) => Normal(y,9).ll(8.0)}.empirical
    print("x: 5.857, 2.867 : ")
    println(breeze.stats.meanAndVariance(xyGz.map(_._1))) // x
    print("y: 6.071, 3.214 : ")
    println(breeze.stats.meanAndVariance(xyGz.map(_._2))) // y
    // Now cond inside for expression...
    val xyz = for {
      x <- Normal(5,4)
      y <- Normal(x,1).cond(y => Normal(y,9).ll(8.0))
    } yield (x,y)
    val xyze = xyz.empirical
    print("x: 5.857, 2.867 : ")
    println(breeze.stats.meanAndVariance(xyze.map(_._1))) // x
    print("y: 6.071, 3.214 : ")
    println(breeze.stats.meanAndVariance(xyze.map(_._2))) // y
    // Now cond inside a deeper for expression...
    val wxyz = for {
      w <- Normal(5,2)
      x <- Normal(w,2)
      y <- Normal(x,1).cond(y => Normal(y,9).ll(8.0))
    } yield (w,x,y)
    val wxyze = wxyz.empirical
    print("w: 5.429, 1.714 : ")
    println(breeze.stats.meanAndVariance(wxyze.map(_._1))) // w
    print("x: 5.857, 2.867 : ")
    println(breeze.stats.meanAndVariance(wxyze.map(_._2))) // x
    print("y: 6.071, 3.214 : ")
    println(breeze.stats.meanAndVariance(wxyze.map(_._3))) // y
    // Now fit...
    val xyzf = for {
      x <- Normal(5,4)
      y <- Normal(x,1)
      z <- Normal(y,9).fit(8.0)
    } yield (x,y,z)
    val xyzfe = xyzf.empirical
    print("x: 5.857, 2.867 : ")
    println(breeze.stats.meanAndVariance(xyzfe.map(_._1))) // x
    print("y: 6.071, 3.214 : ")
    println(breeze.stats.meanAndVariance(xyzfe.map(_._2))) // y
    print("z: 6.071,12.214 : ")
    println(breeze.stats.meanAndVariance(xyzfe.map(_._3))) // z
    // Now fitQ...
    val xyzfq = for {
      x <- Normal(5,4)
      y <- Normal(x,1)
      z <- Normal(y,9).fitQ(8.0)
    } yield (x,y,z)
    val xyzfqe = xyzfq.empirical
    print("x: 5.857, 2.867 : ")
    println(breeze.stats.meanAndVariance(xyzfqe.map(_._1))) // x
    print("y: 6.071, 3.214 : ")
    println(breeze.stats.meanAndVariance(xyzfqe.map(_._2))) // y
    print("z: 8.000, 0.000 : ")
    println(breeze.stats.meanAndVariance(xyzfqe.map(_._3))) // z
    // Simpler fit test
    val yzf = for {
      y <- Normal(5,5)
      z <- Normal(y,9).fit(8.0)
    } yield (y,z)
    val yzfe = yzf.empirical
    print("y: 6.071, 3.214 : ")
    println(breeze.stats.meanAndVariance(yzfe.map(_._1))) // y
    print("z: 6.071,12.214 : ")
    println(breeze.stats.meanAndVariance(yzfe.map(_._2))) // z
    // Simpler fit test - multiple obs
    val yzf2 = for {
      y <- Normal(5,5)
      z <- Normal(y,18).fit(List(6.0,10.0))
    } yield (y,z)
    val yzfe2 = yzf2.empirical
    print("y: 6.071, 3.214 : ")
    println(breeze.stats.meanAndVariance(yzfe2.map(_._1))) // y
    print("z: 6.071,21.214 : ")
    println(breeze.stats.meanAndVariance(yzfe2.map(_._2))) // z
  }

  def example1a = {
    val deep = for {
      w <- Normal(2.0,1.0)
      x <- Normal(w,1)
      y <- Normal(x,2)
      z <- Normal(y,1)
    } yield z
    println("2.0, 5.0 :")
    println(breeze.stats.meanAndVariance(deep.empirical))
  }

  // Normal random sample
  def example2 = {
    val mod = for {
      mu <- Normal(0,100)
      v <- Gamma(1,0.1)
      _ <- Normal(mu,v).fitQ(List(8.0,9,7,7,8,10))
    } yield (mu,v)
    val modEmp = mod.empirical
    print("mu : ")
    println(breeze.stats.meanAndVariance(modEmp map (_._1)))
    print("v : ")
    println(breeze.stats.meanAndVariance(modEmp map (_._2)))
  }

  // Normal random sample - IG on v
  def example2a = {
    val mod = for {
      mu <- Normal(0, 100)
      tau <- Gamma(1, 0.1)
      _ <- Normal(mu, 1.0/tau).fitQ(List(8.0,9,7,7,8,10))
    } yield (mu,tau)
    val modEmp = mod.empirical
    print("mu : ")
    println(breeze.stats.meanAndVariance(modEmp map (_._1)))
    print("tau : ")
    println(breeze.stats.meanAndVariance(modEmp map (_._2)))
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
    example2a
    println("Bye")
  }


}


// eof

