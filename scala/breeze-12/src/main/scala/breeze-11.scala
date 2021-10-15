/*
Stub.scala
Stub for Scala Breeze code
*/

object Stub {

  def main(args: Array[String]): Unit = {

    import breeze.stats.distributions.Rand.VariableSeed.randBasis
    println(breeze.stats.distributions.Poisson(10).sample(5))

    import dev.ludovic.netlib.BLAS
    println(BLAS.getInstance().getClass().getName())
  }

}
