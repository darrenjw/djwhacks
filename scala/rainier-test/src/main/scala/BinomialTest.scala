/*
BinomialTest.scala
Test the binomial likelihood

 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._
import scala.util.Random

object BinomialTest {

  def genBernoulli(p: Double, r: Random): Int =
    if (r.nextDouble < p) 1 else 0

  def genBinomial(n: Int, p: Double, r: Random): Int =
    (1 to n).map(i => genBernoulli(p,r)).sum

  def main(args: Array[String]): Unit = {

    // first simulate some data
    val r = new Random
    val N = 10000
    val n = 10
    val p = 0.2
    val x = (1 to N) map { i => genBinomial(n,p,r) }
    println(x.take(20))

    // now build and fit model
    val model = for {
      p <- Uniform(0, 1).param
      _ <- Binomial(p,n).fit(x)
    } yield (p)

    implicit val rng = RNG.default

    val out = model.sample()
    //val out = model.sample(HMC(5), 1000, 10000)
    println(out.take(10))
    println(s"p (true value $p):")
    println(DensityPlot().plot1D(out).mkString("\n"))

  }

}

// eof
