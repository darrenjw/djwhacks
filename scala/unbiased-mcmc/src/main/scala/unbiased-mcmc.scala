/*
unbiased-mcmc.scala

Messing around with coupling and debiasing of MCMC chains

*/

import breeze.stats.distributions._

object UnbiasedMcmc {

  // Monadic max coupling of two continuous distributions
  def couple[T](p: ContinuousDistr[T], q: ContinuousDistr[T]): Rand[(T,T)] = {
    def ys: Rand[T] = for {
      y <- q
      w <- Uniform(0, q.pdf(y))
      ay <- if (w > p.pdf(y)) Rand.always(y) else ys
    } yield ay // TODO: use tailRecM to make tail recursive
    val pair = for {
      x <- p
      w <- Uniform(0, p.pdf(x))
    } yield (w <= q.pdf(x), x)
    pair flatMap {
      case (b,x) => if (b) Rand.always((x,x)) else (ys map (y => (x,y)))
    }
  }

  // Monadic coupling of a Metropolis kernel for target logPi
  def coupledMetKernel[T](q: T => ContinuousDistr[T])(logPi: T =>
      Double)(x: (T,T)): Rand[(T,T)] = for {
    p <- couple(q(x._1), q(x._2))
    //p <- for (q1 <- q(x._1); q2 <- q(x._2)) yield (q1,q2) // uncoupled proposal
    u <- Uniform(0,1)
    n1 = if (math.log(u) < logPi(p._1) - logPi(x._1)) p._1 else x._1
    n2 = if (math.log(u) < logPi(p._2) - logPi(x._2)) p._2 else x._2
  } yield (n1,n2)


  def couplingTest: Unit = {
    //val c = couple(Gamma(10,0.1), Gamma(10,0.1))
    val c = couple(Gamma(10,0.1), Gamma(20,0.1))
    //val c = couple(Gaussian(5,2), Gaussian(4,3))
    //val c = couple(Gaussian(0,1),Gaussian(1,1))
    //val c = couple(Uniform(0,2),Uniform(1,4))
    val cs = c.sample(10000)
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

  def metropTest: Unit = {
    val chain = MarkovChain.metropolis(0.0,
      (x: Double) => Uniform(x-0.5, x+0.5))(
      x => Gaussian(0.0, 1.0).logPdf(x)).
      steps.
      drop(1).
      take(10000).
      toArray
    import breeze.plot._
    import breeze.linalg._
    val fig = Figure("Metropolis chain")
    val p0 = fig.subplot(0)
    p0 += plot(linspace(1, chain.length, chain.length), chain)
  }

  def coupledMetropTest: Unit = {
    val chain = MarkovChain((5.0,-5.0))(coupledMetKernel((x: Double) => Uniform(x-0.5, x+0.5))(x => Gaussian(0.0,1.0).logPdf(x))).
      steps.
      sliding(2).
      takeWhile(ps => ps.head._1 != ps.head._2).
      map(ps => ps.tail.head).
      toArray
    val x = chain map (_._1)
    val y = chain map (_._2)
    import breeze.plot._
    import breeze.linalg._
    val fig = Figure("Pair of coupled Metropolis chains")
    val p0 = fig.subplot(0)
    p0 += plot(linspace(1, x.length, x.length), x)
    p0 += plot(linspace(1, y.length, y.length), y)
  }



  def main(args: Array[String]): Unit = {
    //couplingTest
    //metropTest
    coupledMetropTest
  }

}
