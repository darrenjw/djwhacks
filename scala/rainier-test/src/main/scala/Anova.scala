/*
MyAnova.scala

Try doing a one-way ANOVA with random effects model using Rainier

 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._

object MyAnova {

  import scala.language.higherKinds
  trait Thinnable[F[_]] {
    def thin[T](f: F[T], th: Int): F[T]
  }

  implicit class ThinnableSyntax[T,F[T]](value: F[T]) {
    def thin(th: Int)(implicit inst: Thinnable[F]): F[T] =
      inst.thin(value,th)
  }

  implicit val streamThinnable: Thinnable[Stream] =
    new Thinnable[Stream] {
      def thin[T](s: Stream[T],th: Int): Stream[T] = {
        val ss = s.drop(th-1)
        if (ss.isEmpty) Stream.empty else
                                       ss.head #:: thin(ss.tail, th)
      }
    }



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
    //val out = model.sample(Walkers(1000), 100000, its)
    val out = model.sample(HMC(5), 1000000, its*thin, thin)
    //val outs = model.toStream(HMC(5),1000000)
    //println("Warmed up. Now sampling.")
    //val out = outs.thin(thin).take(its).map(_()).toList
    println("Sampling finished.")

    println(out.take(10))
    println("Iterates (requested): " + its)
    println("Iterates (actual): " + out.length)

    import breeze.plot._
    import breeze.linalg._
    val fig = Figure("MCMC Diagnostics")
    fig.height = 1000
    fig.width = 1400
    val p0 = fig.subplot(5,2,0)
    p0 += plot(linspace(1,its,its),out map (_("Mu")))
    p0.title = s"mu (true value $mu):"
    val p1 = fig.subplot(5,2,1)
    p1 += hist(out map (_("Mu")))
    p1 += plot(linspace(mu,mu,2),linspace(0,p1.ylim._2,2))
    val p2 = fig.subplot(5,2,2)
    p2 += plot(linspace(1,its,its),out map (_("sigE")))
    p2.title = s"sigE (true value $sigE):"
    val p3 = fig.subplot(5,2,3)
    p3 += hist(out map (_("sigE")))
    p3 += plot(linspace(sigE,sigE,2),linspace(0,p3.ylim._2,2))
    val p4 = fig.subplot(5,2,4)
    p4 += plot(linspace(1,its,its),out map (_("sigD")))
    p4.title = s"sigD (true value $sigD):"
    val p5 = fig.subplot(5,2,5)
    p5 += hist(out map (_("sigD")))
    p5 += plot(linspace(sigD,sigD,2),linspace(0,p5.ylim._2,2))
    val p6 = fig.subplot(5,2,6)
    p6 += plot(out map (_("Mu")),out map (_("sigE")),'.')
    p6.xlabel = "Mu"
    p6.ylabel = "sigE"
    p6.title = "sigE against Mu"
    p6 += plot(linspace(mu,mu,2),linspace(p6.ylim._1,p6.ylim._2,2))
    p6 += plot(linspace(p6.xlim._1,p6.xlim._2,2),linspace(sigE,sigE,2))
    val p7 = fig.subplot(5,2,7)
    p7 += plot(out map (_("Mu")),out map (_("sigD")),'.')
    p7.xlabel = "Mu"
    p7.ylabel = "sigD"
    p7.title = "sigD against Mu"
    p7 += plot(linspace(mu,mu,2),linspace(p7.ylim._1,p7.ylim._2,2))
    p7 += plot(linspace(p7.xlim._1,p7.xlim._2,2),linspace(sigD,sigD,2))
    val p8 = fig.subplot(5,2,8)
    p8 += plot(out map (_("sigE")),out map (_("sigD")),'.')
    p8.xlabel = "sigE"
    p8.ylabel = "sigD"
    p8.title = "sigD against sigE"
    p8 += plot(linspace(sigE,sigE,2),linspace(p8.ylim._1,p8.ylim._2,2))
    p8 += plot(linspace(p8.xlim._1,p8.xlim._2,2),linspace(sigD,sigD,2))



  }

}

// eof
