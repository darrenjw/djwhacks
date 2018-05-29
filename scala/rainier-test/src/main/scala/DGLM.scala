/*
MyDGLM.scala

Try doing a DGLM - AR(1) latent state and Poisson observations


 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._

object MyDGLM {

  implicit val rng = RNG.default

  // TODO: Spectacularly bad way of generating Poisson RVs...
  def genPois(lambda: Double): Int =
    RandomVariable(Poisson(lambda).generator).sample().head

  def main(args: Array[String]): Unit = {

    // first simulate some data from a DGLM model
    val r = new scala.util.Random
    val n = 250 // time points
    val mu = 1.0 // AR(1) mean
    val a = 0.95 // auto-regressive parameter
    val sig = 0.5 // AR(1) SD
    val state = Stream
      .iterate(0.0)(x => mu + (x - mu) * a + sig * r.nextGaussian)
      .take(n)
      .toVector
    val obs = state map (s => genPois(math.exp(s)))

    // build and fit model
    val prior = for {
      mu <- Normal(0, 100).param
      a <- Normal(1, 0.2).param
      sig <- LogNormal(0, 10).param
      sp <- Normal(0, 100).param
    } yield Map("mu" -> mu, "a" -> a, "sig" -> sig, "SP" -> sp)

    def addTimePoint(current: RandomVariable[Map[String, Real]], i: Int) =
      for {
        map <- current
        os = if (i == 0) map("SP") else map(f"S$i%03d")
        ni = i + 1
        ns <- Normal(map("mu") + map("a") * (os - map("mu")), map("sig")).param
        _ <- Poisson(ns.exp).fit(obs(i))
      } yield map.updated(f"S$ni%03d", ns)

    val fullModel = (0 until n).foldLeft(prior)(addTimePoint(_, _))

    val model = for {
      map <- fullModel
    } yield
      Map(
        "mu" -> map("mu"),
        "a" -> map("a"),
        "sig" -> map("sig"),
        "SP" -> map("SP")
      )

    println("Model built. Sampling now...")
    //val out = model.sample()
    //val out = model.sample(Walkers(100), 1000, 2000)
    val out = model.sample(HMC(10), 100, 200)
    //println("Sampling finished.")

    println("Iterates: " + out.length)
    println("First 20:")
    println(out.take(20))
    println(s"Mu (true value $mu):")
    println(DensityPlot().plot1D(out map (_("mu"))).mkString("\n"))
    println(s"a (true value $a):")
    println(DensityPlot().plot1D(out map (_("a"))).mkString("\n"))
    println(s"sig (true value $sig):")
    println(DensityPlot().plot1D(out map (_("sig"))).mkString("\n"))
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
