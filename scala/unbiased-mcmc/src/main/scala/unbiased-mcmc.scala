/*
unbiased-mcmc.scala

Messing around with coupling and debiasing of MCMC chains

*/

object UnbiasedMcmc {

  import breeze.stats.distributions._

  // Monadic coupling of two continuous distributions
  def couple[T](p: ContinuousDistr[T], q: ContinuousDistr[T]): Rand[(T,T)] = {
    def ys: Rand[T] = for {
      y <- q
      w <- Uniform(0, q.pdf(y))
      ayd = if (w > p.pdf(y)) Rand.always(y) else ys
      ay <- ayd
    } yield ay
    for {
      x <- p
      w <- Uniform(0, p.pdf(x))
      y <- ys
    } yield if (w < q.pdf(x)) (x,x) else (x,y)
  }



  def main(args: Array[String]): Unit = {

    // val c = couple(Gamma(10,5), Gamma(20,2.5))
    val c = couple(Gaussian(5,2), Gaussian(4,3))
    val cs = c.sample(1000)
    val x = cs map (_._1)
    val y = cs map (_._2)

    import breeze.plot._
    val fig = Figure("Coupling")
    val p0 = fig.subplot(1,3,0)
    p0 += plot(x,y,'.')
    val p1 = fig.subplot(1,3,1)
    p1 += hist(x)
    val p2 = fig.subplot(1,3,2)
    p2 += hist(y)

  }

}
