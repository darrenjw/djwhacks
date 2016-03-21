
object BreezeMcmc {

  import breeze.stats.distributions.MarkovChain._
  import breeze.stats.distributions.{Gamma, Gaussian}
  import breeze.numerics._
  import breeze.linalg._

  def main(args: Array[String]): Unit = {
    val mc = metropolisHastings(1.0, (x: Double) =>
      new Gaussian(x, 1.0))(x => Gamma(2.0, 1.0/3).logPdf(x))
    val it = mc.steps
    val its = it.take(100000).toArray
    val itsv = DenseVector[Double](its)
    println(min(itsv))
    println(sum(itsv) / itsv.length)
    println("This doesn't work! Bugs in MH Kernel implementation...")
    // Try Stuccio's stuff instead?!
    import breeze.stats.mcmc.ArbitraryMetropolisHastings
    val mh = ArbitraryMetropolisHastings(Gamma(2.0, 1.0/3).logPdf, 
      (x: Double) => Gaussian(x, 1.0), 
      (x: Double, xp: Double) => Gaussian(x, 1.0).logPdf(xp), 1.0)
    val sit = mh.samples
    val its2=sit.take(100000).toArray
    val itsv2=DenseVector[Double](its2)
    println(min(itsv2))
    println(sum(itsv2)/itsv2.length)
    println("At least this one seems to work")
  }

}
