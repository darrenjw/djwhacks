/*
DLM.scala

Try doing a DLM - AR(1) latent state and Gaussian observations

 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._

object DLM {

  def main(args: Array[String]): Unit = {

    // first simulate some data from a DLM model
    val r = new scala.util.Random
    val n = 250 // time points
    //val n = 1000 // time points
    val mu = 3.0 // AR(1) mean
    val a = 0.95 // auto-regressive parameter
    val sig = 2.0 // AR(1) SD
    val sigD = 3.0 // observational SD
    val state = Stream
      .iterate(0.0)(x => mu + (x - mu) * a + sig * r.nextGaussian)
      .take(n)
      .toVector
    val obs = state map (_ + sigD * r.nextGaussian)

    // build and fit model
    val prior = for {
      mu <- Normal(5, 10).param
      a <- Normal(1, 0.2).param
      sig <- LogNormal(0, 10).param
      sigD <- LogNormal(0, 10).param
      sp <- Normal(0, 100).param
    } yield Map("mu" -> mu, "a" -> a, "sig" -> sig, "sigD" -> sigD, "SP" -> sp)

    def addTimePoint(current: RandomVariable[Map[String, Real]], i: Int) =
      for {
        map <- current
        os = if (i == 0) map("SP") else map(f"S$i%03d")
        ni = i + 1
        ns <- Normal(map("mu") + map("a") * (os - map("mu")), map("sig")).param
        _ <- Normal(ns, map("sigD")).fit(obs(i))
      } yield map.updated(f"S$ni%03d", ns)

    val fullModel = (0 until n).foldLeft(prior)(addTimePoint(_, _))

    val model = for {
      map <- fullModel
    } yield
      Map(
        "mu" -> map("mu"),
        "a" -> map("a"),
        "sig" -> map("sig"),
        "sigD" -> map("sigD"),
        "SP" -> map("SP")
      )

    implicit val rng = RNG.default

    println("Model built. Sampling now...")
    //val out = model.sample()
    val out = model.sample(Walkers(100), 1000, 2000)
    //val out = model.sample(HMC(10), 1000, 2000)
    println("Sampling finished.")

    println("Iterates: " + out.length)
    println("First 20:")
    println(out.take(20))
    println(s"Mu (true value $mu):")
    println(DensityPlot().plot1D(out map (_("mu"))).mkString("\n"))
    println(s"a (true value $a):")
    println(DensityPlot().plot1D(out map (_("a"))).mkString("\n"))
    println(s"sig (true value $sig):")
    println(DensityPlot().plot1D(out map (_("sig"))).mkString("\n"))
    println(s"sigD (true value $sigD):")
    println(DensityPlot().plot1D(out map (_("sigD"))).mkString("\n"))
    println("Scatter of sig against mu")
    println(
      DensityPlot()
        .plot2D(out map { r =>
          (r("mu"), r("sig"))
        })
        .mkString("\n"))
    //Report.printReport(model, Emcee(5000, 2000, 200))

  }

}

// eof
