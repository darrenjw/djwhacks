/*
Failing case
 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._

object Failing {

  def main(args: Array[String]): Unit = {

    // first simulate some data from an ANOVA model
    val r = new scala.util.Random(0)
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
      sigE <- LogNormal(1, 5).param
    } yield Map("Mu" -> mu, "sigD" -> sigD, "sigE" -> sigE)

    def addGroup(current: Map[String, Real], i: Int) = for {
        gm <- Normal(current("Mu"), current("sigE")).param
        _ <- Normal(gm, current("sigD")).fit(data(i))
      } yield gm 

    val model = for {
      current <- prior
      _ <- RandomVariable.traverse((0 until n) map (addGroup(current, _)))
    } yield current

    implicit val rng = ScalaRNG(3)
    println("Model built. Sampling now...")
    val its = 10000
    val thin = 1000
    val out = model.sample(HMC(5), 1000000, its*thin, thin)
    println("Sampling finished.")

  }

}

// eof
