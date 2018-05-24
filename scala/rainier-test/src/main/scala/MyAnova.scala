/*
MyAnova.scala

Try doing a one-way ANOVA with random effects model using Rainier


 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._
//import com.stripe.rainier.report._

object MyAnova {

  def main(args: Array[String]): Unit = {

    // first simulate some data from an ANOVA model
    val r = new scala.util.Random
    val n = 50 // groups
    val N = 250 // obs per group
    val mu = 5.0 // overall mean
    val sigE = 2.0 // random effect SD
    val sigD = 3.0 // obs SD
    val effects = Vector.fill(n)(sigE * r.nextGaussian)
    val data = effects map (e => Vector.fill(N)(mu + e + sigD * r.nextGaussian))

    // build and fit model
    val prior = for {
      mu <- Normal(0, 100).param
      sigD <- LogNormal(0, 10).param
      sigE <- LogNormal(0, 10).param
    } yield Map("Mu" -> mu, "sigD" -> sigD, "sigE" -> sigE)

    def addGroup(current: RandomVariable[Map[String, Real]], i: Int) =
      for {
        map <- current
        gm <- Normal(map("Mu"), map("sigE")).param
        _ <- Normal(gm, map("sigD")).fit(data(i))
      } yield map //.updated(f"M$i%02d", gm)

    val model = (0 until n).foldLeft(prior)(addGroup(_, _))

    implicit val rng = RNG.default

    println("Model built. Sampling now...")
    //val out = model.sample()
    //val out = model.sample(Emcee(5000, 2000, 200))
    val out =
      model.sample(HMC(10), 10000, 20000)
    println("Sampling finished.")

    println(out.take(20))
    println("Iterates: " + out.length)
    println(s"Mu (true value $mu):")
    println(DensityPlot().plot1D(out map (_("Mu"))).mkString("\n"))
    println(s"sigE (true value $sigE):")
    println(DensityPlot().plot1D(out map (_("sigE"))).mkString("\n"))
    println(s"sigD (true value $sigD):")
    println(DensityPlot().plot1D(out map (_("sigD"))).mkString("\n"))
    println("Scatter of sigE against Mu")
    println(
      DensityPlot()
        .plot2D(out map { r =>
          (r("Mu"), r("sigE"))
        })
        .mkString("\n"))
    println("Scatter of sigD against Mu")
    println(
      DensityPlot()
        .plot2D(out map { r =>
          (r("Mu"), r("sigD"))
        })
        .mkString("\n"))
    println("Scatter of sigE against sigD")
    println(
      DensityPlot()
        .plot2D(out map { r =>
          (r("sigD"), r("sigE"))
        })
        .mkString("\n"))
    //new Report(List(out)).printReport()
    //Report.printReport(model,
    //                 Hamiltonian(10000, 2000, 100, 10, SampleHMC, 4, 0.001))
    //Report.printReport(model, Emcee(5000, 2000, 200))

  }

}

// eof
