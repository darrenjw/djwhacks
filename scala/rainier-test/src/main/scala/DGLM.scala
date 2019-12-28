/*
DGLM.scala

Try doing a DGLM - AR(1) latent state and Poisson observations


 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._

object DGLM {

  implicit val rng = RNG.default

  // TODO: *Spectacularly* bad way of generating Poisson RVs...
  def genPois(lambda: Double): Int =
    RandomVariable(Poisson(lambda).generator).sample().head

  def main(args: Array[String]): Unit = {

    // first simulate some data from a DGLM model
    val r = new scala.util.Random
    val n = 50 // time points
    val mu = 2.0 // AR(1) mean
    val a = 0.95 // auto-regressive parameter
    val sig = 0.5 // AR(1) SD
    val state = Stream
      .iterate(0.0)(x => mu + (x - mu) * a + sig * r.nextGaussian)
      .take(n)
      .toVector
    val obs = state map (s => genPois(math.exp(s)))
    println("Simulated observations:")
    println(obs)

    case class Static(mu: Real, a: Real, sig: Real)

    // build and fit model
    val prior = for {
      mu <- Normal(1, 10).param
      a <- Normal(1, 0.2).param
      sig <- LogNormal(0, 2).param
      sp <- Normal(0, 10).param
    } yield (Static(mu, a, sig), List(sp))

    def addTimePoint(current: RandomVariable[(Static, List[Real])], i: Int) =
      for {
        tup <- current
        static = tup._1
        states = tup._2
        os = states.head
        ni = i + 1
        ns <- Normal(static.mu + static.a * (os - static.mu), static.sig).param
        _ <- Poisson(ns.exp).fit(obs(i))
      } yield (static, ns :: states)

    val fullModel = (0 until n).foldLeft(prior)(addTimePoint(_, _))

    val model = for {
      tup <- fullModel
      static = tup._1
      states = tup._2
    } yield
      Map(
        "mu" -> static.mu,
        "a" -> static.a,
        "sig" -> static.sig,
        "SP" -> states.reverse.head
      )

    println("Model built. Sampling now...")
    val thin = 100
    val out = model.sample(HMC(5), 10000, 10000*thin, thin)

    // some diagnostic plots
    import com.cibo.evilplot.geometry.Extent
    import com.stripe.rainier.plot.EvilTracePlot._

    println("Iterates: " + out.length)
    render(traces(out,
                  truth = Map("mu" -> mu,
                              "a" -> a,
                              "sig" -> sig,
                              "SP" -> state(0))),
           "traceplots.png",
           Extent(1200, 1400))
    render(pairs(out,
                 truth = Map("mu" -> mu,
                             "a" -> a,
                             "sig" -> sig,
                             "SP" -> state(0))),
           "pairs.png")
    println("Diagnostic plots written to disk")


  }

}

// eof
