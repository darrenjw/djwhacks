/*
MyLogReg.scala

Try doing a logistic regression model using Rainier

 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._
//import com.stripe.rainier.report._

case class Bernoulli(p: Real) extends Distribution[Int] {

  def logDensity(b: Int): Real = {
    p.log * b + (Real.one - p).log * (1 - b)
  }

  val generator = Generator.from { (r, n) =>
    val pd = n.toDouble(p)
    val u = r.standardUniform
    if (u < pd) 1 else 0
  }

}

object MyLogReg {

  def main(args: Array[String]): Unit = {

    // first simulate some data from a logistic regression model
    val r = new scala.util.Random
    val N = 2000
    val beta0 = 0.1
    val beta1 = 0.3
    val x = (1 to N) map { i =>
      3.0 * r.nextGaussian
    }
    val theta = x map { xi =>
      beta0 + beta1 * xi
    }
    def expit(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
    val p = theta map expit
    val y = p map (pi => if (r.nextDouble < pi) 1 else 0)

    // now build and fit model
    val model = for {
      beta0 <- Normal(0, 5).param
      beta1 <- Normal(0, 5).param
      _ <- Predictor
        .from { x: Double =>
          {
            val theta = beta0 + beta1 * x
            val p = Real(1.0) / (Real(1.0) + (Real(0.0) - theta).exp)
            Bernoulli(p)
          }
        }
        .fit(x zip y)
    } yield (beta0, beta1)

    implicit val rng = RNG.default

    //val out = model.sample()
    //val out = model.sample(Emcee(5000, 2000, 200))
    val out =
      model.sample(HMC(10), 10000, 20000)
    println(out.take(10))
    println(s"b0 (true value $beta0):")
    println(DensityPlot().plot1D(out map (_._1)).mkString("\n"))
    println(s"b1 (true value $beta1):")
    println(DensityPlot().plot1D(out map (_._2)).mkString("\n"))
    println("b1 against b0:")
    println(DensityPlot().plot2D(out).mkString("\n"))

    //Report.printReport(model map {case (b0, b1) => Map("b0" -> b0, "b1" -> b1)},
    //                   Emcee(5000, 2000, 200))

  }

}

// eof
