/*
Crash.scala
Crashes Rainier
 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._

object Crash {

  def main(args: Array[String]): Unit = {

    // first simulate some data from a DLM model
    val r = new scala.util.Random(0)
    val n = 50 // time points
    val mu = 3.0 // AR(1) mean
    val a = 0.95 // auto-regressive parameter
    val sig = 1.0 // AR(1) SD
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

    implicit val rng = ScalaRNG(4)

    println("Model built. Sampling now...")
    val out = model.sample(HMC(5), 5000, 10000)
    println("Sampling finished.")

  }

}

// eof
