import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._

object DLM3 {

  def main(args: Array[String]): Unit = {

    // first simulate some data from a DLM model
    implicit val rng = ScalaRNG(1)
    val n = 60 // number of observations/time points
    val mu = 3.0 // AR(1) mean
    val a = 0.95 // auto-regressive parameter
    val sig = 0.2 // AR(1) SD
    val sigD = 3.0 // observational SD
    val state = Stream.
      iterate(0.0)(x => mu + (x - mu) * a + sig * rng.standardNormal).
      take(n).toVector
    val obs = state.map(_ + sigD * rng.standardNormal)

    // build and fit model
    case class Static(mu: Real, a: Real, sig: Real, sigD: Real)

    val prior = for {
      mu <- Normal(0, 10).param
      a <- Normal(1, 0.1).param
      sig <- Gamma(2,1).param
      sigD <- Gamma(2,2).param
      sp <- Normal(0, 50).param
    } yield (Static(mu, a, sig, sigD), List(sp))

    def addTimePoint(current: RandomVariable[(Static, List[Real])],
                     datum: Double): RandomVariable[(Static, List[Real])] =
      for {
        tup <- current
        static = tup._1
        states = tup._2
        os = states.head
        ns <- Normal(((Real.one - static.a) * static.mu) + (static.a * os),
                     static.sig).param
        _ <- Normal(ns, static.sigD).fit(datum)
      } yield (static, ns :: states)

    val fullModel = obs.foldLeft(prior)(addTimePoint(_, _))

    val model = for {
      tup <- fullModel
      static = tup._1
      states = tup._2
    } yield
      Map("mu" -> static.mu,
          "a" -> static.a,
          "sig" -> static.sig,
          "sigD" -> static.sigD,
          "SP" -> states.reverse.head)

    // sampling
    println("Model built. Sampling now (will take a long time)...")
    val thin = 500
    val out = model.sample(HMC(3), 100000, 10000 * thin, thin)
    println("Sampling finished.")

    // some diagnostic plots
    import com.cibo.evilplot.geometry.Extent
    import com.stripe.rainier.plot.EvilTracePlot._

    val truth = Map("mu" -> mu, "a" -> a, "sigD" -> sigD,
      "sig" -> sig, "SP" -> state(0))
    render(traces(out, truth), "traceplots.png",
           Extent(1200, 1400))
    render(pairs(out, truth), "pairs.png")
    println("Diagnostic plots written to disk")

  }

}
