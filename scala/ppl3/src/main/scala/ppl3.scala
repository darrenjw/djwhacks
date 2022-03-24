/*
ppl3.scala
Stub for Scala Cats code
*/

import cats.*
import cats.implicits.*
import breeze.stats.{distributions => bdist}
import breeze.linalg.DenseVector
import breeze.stats.distributions.Rand.VariableSeed.randBasis

object PPL3:

  opaque type NumParticles = Int
  object NumParticles:
    def apply(np: Int): NumParticles = np
  given NumParticles = 2000

  case class Particle[T](v: T, lw: Double): // value and log-weight
    def map[S](f: T => S): Particle[S] = Particle(f(v), lw)
    def flatMap[S](f: T => Particle[S]): Particle[S] =
      val ps = f(v)
      Particle(ps.v, lw + ps.lw)

  given Monad[Particle] = new Monad[Particle]:
    def pure[T](t: T): Particle[T] = Particle(t, 0.0)
    def flatMap[T,S](pt: Particle[T])(f: T => Particle[S]): Particle[S] = pt.flatMap(f)
    def tailRecM[T,S](t: T)(f: T => Particle[Either[T,S]]): Particle[S] = ???

  trait Prob[T]:
    lazy val particles: Vector[Particle[T]]
    def draw: Particle[T]
    def mapP[S](f: T => Particle[S]): Prob[S] = Empirical(particles map (_ flatMap f))
    def map[S](f: T => S): Prob[S] = mapP(v => Particle(f(v), 0.0))
    def flatMap[S](f: T => Prob[S]): Prob[S] = mapP(f(_).draw)
    def resample(using N: NumParticles): Prob[T] =
      val lw = particles map (_.lw)
      val mx = lw reduce (math.max(_,_))
      val rw = lw map (lwi => math.exp(lwi - mx))
      val law = mx + math.log(rw.sum/(rw.length))
      val ind = bdist.Multinomial(DenseVector(rw.toArray)).sample(N)
      val newParticles = ind map (i => particles(i))
      Empirical(newParticles.toVector map (pi => Particle(pi.v, law)))
    def cond(ll: T => Double): Prob[T] = mapP(v => Particle(v, ll(v)))
    def empirical: Vector[T] = resample.particles.map(_.v)

  given Monad[Prob] = new Monad[Prob]:
    def pure[T](t: T): Prob[T] = Empirical(Vector(Particle(t, 0.0)))
    def flatMap[T,S](pt: Prob[T])(f: T => Prob[S]): Prob[S] = pt.flatMap(f)
    def tailRecM[T,S](t: T)(f: T => Prob[Either[T,S]]): Prob[S] =
      flatMap(f(t)) {
        case Right(b) => pure(b)
        case Left(nextA) => tailRecM(nextA)(f)
      }

  // TODO: factor out common logic for resample and draw

  case class Empirical[T](samples: Vector[Particle[T]]) extends Prob[T]:
    lazy val particles = samples
    lazy val lw = particles map (_.lw)
    lazy val mx = lw reduce (math.max(_,_))
    lazy val rw = lw map (lwi => math.exp(lwi - mx))
    lazy val law = mx + math.log(rw.sum/(rw.length))
    lazy val dist = bdist.Multinomial(DenseVector(rw.toArray))
    def draw: Particle[T] =
      val idx = dist.draw()
      Particle(particles(idx).v, law)

  // TODO: should wrap in a case object? Or just part of Dist??
  def unweighted[T](ts: Vector[T], lw: Double = 0.0): Prob[T] =
    Empirical(ts map (Particle(_, lw)))

  trait Dist[T] extends Prob[T]:
    def ll(obs: T): Double
    def ll(obs: Seq[T]): Double = obs map (ll) reduce (_+_)
    def fit(obs: Seq[T]): Prob[T] = mapP(v => Particle(v, ll(obs)))
    def fitQ(obs: Seq[T]): Prob[T] = Empirical(Vector(Particle(obs.head, ll(obs))))
    def fit(obs: T): Prob[T] = fit(List(obs))
    def fitQ(obs: T): Prob[T] = fitQ(List(obs))

  case class Normal(mu: Double, v: Double)(using N: NumParticles) extends Dist[Double]:
    lazy val particles = unweighted(bdist.Gaussian(mu, math.sqrt(v)).
      sample(N).toVector).particles
    def draw = Particle(bdist.Gaussian(mu, math.sqrt(v)).draw(), 0.0)
    def ll(obs: Double) = bdist.Gaussian(mu, math.sqrt(v)).logPdf(obs)

  case class Gamma(a: Double, b: Double)(using N: NumParticles) extends Dist[Double]:
    lazy val particles = unweighted(bdist.Gamma(a, 1.0/b).
      sample(N).toVector).particles
    def draw = Particle(bdist.Gamma(a, 1.0/b).draw(), 0.0)
    def ll(obs: Double) = bdist.Gamma(a, 1.0/b).logPdf(obs)

  case class Poisson(mu: Double)(using N: NumParticles) extends Dist[Int]:
    lazy val particles = unweighted(bdist.Poisson(mu).
      sample(N).toVector).particles
    def draw = Particle(bdist.Poisson(mu).draw(), 0.0)
    def ll(obs: Int) = bdist.Poisson(mu).logProbabilityOf(obs)



object Example:

  import PPL3.*
  import breeze.stats.{meanAndVariance => meanVar}

  // Deep monadic binding issues
  def example1 =
    println("binding with for")
    val prior1 = for
      x <- Normal(0,1)
      y <- Gamma(1,1)
      z <- Poisson(10)
    yield (x,y,z)
    println(meanVar(prior1.empirical.map(_._2)))
    println("binding with flatMap")
    val prior2 =
      Normal(0,1) flatMap {x =>
        Gamma(1,1) flatMap {y =>
          Poisson(10) map {z =>
            (x,y,z)}}}
    println(meanVar(prior2.empirical.map(_._2)))

  def example2 =
    println("tupling")
    val prior3 = Applicative[Prob].tuple3(Normal(0,1), Gamma(1,1), Poisson(10))
    println(meanVar(prior3.empirical.map(_._2)))
    print("done")

  // Poisson DGLM
  def example3 =

    val data = List(2,1,0,2,3,4,5,4,3,2,1)

    val prior = for {
      w <- Gamma(1, 1)
      state0 <- Normal(0.0, 2.0)
    } yield (w, List(state0))
    
    def addTimePointSimple(current: Prob[(Double, List[Double])],
      obs: Int): Prob[(Double, List[Double])] = {
      println(s"Conditioning on observation: $obs")
      val updated = for {
        tup <- current
        (w, states) = tup
        os = states.head
        ns <- Normal(os, w)
        _ <- Poisson(math.exp(ns)).fitQ(obs)
      } yield (w, ns :: states)
      updated.resample
    }

    def addTimePoint(current: Prob[(Double, List[Double])],
      obs: Int): Prob[(Double, List[Double])] = {
      println(s"Conditioning on observation: $obs")
      val predict = for {
        tup <- current
        (w, states) = tup
        os = states.head
        ns <- Normal(os, w)
      }
      yield (w, ns :: states)
      val updated = for {
        tup <- predict
        (w, states) = tup
        st = states.head
        _ <- Poisson(math.exp(st)).fitQ(obs)
      } yield (w, states)
      updated.resample
    }

    val mod = data.foldLeft(prior)(addTimePoint(_,_)).empirical
    print("w  : ")
    println(meanVar(mod map (_._1)))
    print("s0 : ")
    println(meanVar(mod map (_._2.reverse.head)))
    print("sN : ")
    println(meanVar(mod map (_._2.head)))





  // Main runner method - program entry point
  @main def run =
    given NumParticles = NumParticles(1000) // TODO: ignored!
    example3
