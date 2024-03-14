/*
bridge.scala
HMC for diffusion bridge sampling
*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import breeze.linalg.{Vector => BVec, *}
import breeze.numerics.*
import breeze.stats.distributions.Rand.VariableSeed.randBasis
import breeze.stats.distributions.{Gaussian, Uniform}

import annotation.tailrec

// case class to represent a bridge with fixed end points
case class Bridge[A](c: Int, br: Vector[A]):
  val m = br.length + 1
  def apply(i: Int): A = br(i)
  def extract: A = br(c)
  def map[B](f: A => B): Bridge[B] = Bridge(c, br map f)
  //def coflatMap[B](f: Bridge[A] => B): Bridge[B] = Bridge(
  //  c, (0 until (m-1)).toVector.map(i => f(Bridge(i, br))))
  //def left: Bridge[A] = Bridge(c-1, br)
  //def right: Bridge[A] = Bridge(c+1, br)

case object Bridge:
  def apply(leftFP: Double, rightFP: Double, m: Int): Bridge[Double] =
    Bridge(0, linspace(leftFP, rightFP, m-1).data.toVector)

// Provide evidence that Bridge is a Cats Apply
given Apply[Bridge] with
  def map[A,B](ba: Bridge[A])(f: A => B): Bridge[B] = ba.map(f)
  def ap[A, B](ff: Bridge[A=>B])(fa: Bridge[A]): Bridge[B] =
    Bridge(ff.c, (ff.br zip fa.br).map((ffi, fai) => ffi(fai)))

// Provide evidence that Bridge is a Cats Reducible
given Reducible[Bridge] with
  def foldLeft[A, B](fa: Bridge[A], b: B)(f: (B, A) => B): B =
    fa.br.foldLeft(b)(f)
  def foldRight[A, B](fa: Bridge[A], lb: Eval[B])(f: (A, Eval[B]) => Eval[B]): Eval[B] =
    fa.br.foldRight(lb)(f)
  def reduceLeftTo[A, B](fa: Bridge[A])(f: A => B)(g: (B, A) => B): B =
    fa.br.tail.foldLeft(f(fa.br.head))(g)
  def reduceRightTo[A, B](fa: Bridge[A])(f: A => B)(g: (A, Eval[B]) => Eval[B]): Eval[B] =
    fa.br.init.foldRight(Eval.later(f(fa.br.last)))(g)
  // optionally override reduce
  def reduce[A](fa: Bridge[A])(f: (A, A) => A): A = fa.br.reduce(f)


// Thinnable typeclass and instance for LazyLists
trait Thinnable[F[_]]:
  extension [T](ft: F[T])
    def thin(th: Int): F[T]

given Thinnable[LazyList] with
  extension [T](s: LazyList[T])
    def thin(th: Int): LazyList[T] =
      val ss = s.drop(th-1)
      if (ss.isEmpty) LazyList.empty else
        ss.head #:: ss.tail.thin(th)

object Kernels:

  // a MH kernel as needed for HMC
  def mhKern[S](
      logPost: S => Double, rprop: S => S,
      dprop: (S, S) => Double = (n: S, o: S) => 1.0
    ): (S) => S =
      val r = Uniform(0.0,1.0)
      x0 =>
        val x = rprop(x0)
        val ll0 = logPost(x0)
        val ll = logPost(x)
        val a = ll - ll0 + dprop(x0, x) - dprop(x, x0)
        if (math.log(r.draw()) < a) x else x0

  // a HMC kernel
  def hmcKernel[F[_]: Apply](lpi: F[Double] => Double, glpi: F[Double] => F[Double],
      eps: Double = 1e-4, l: Int = 10)(using Reducible[F]): F[Double] => F[Double] =
    def add(p: F[Double], q: F[Double]): F[Double] = (p product q) map ((pi, qi) => pi + qi)
    def scale(s: Double, p: F[Double]): F[Double] = p map (pi => s * pi)
    def leapf(q: F[Double], p: F[Double]): (F[Double], F[Double]) =
      @tailrec def go(q0: F[Double], p0: F[Double], l: Int): (F[Double], F[Double]) =
        val q = add(q0, scale(eps, p0))
        val p = if (l > 1)
          add(p0, scale(eps, glpi(q)))
        else
          add(p0, scale(0.5*eps, glpi(q)))
        if (l == 1)
          (q, p)
        else
          go(q, p, l-1)
      go(q, add(p, scale(0.5*eps, glpi(q))), l)
    def alpi(x: (F[Double], F[Double])): Double =
      val (q, p) = x
      lpi(q) - 0.5*(p.map(pi => pi*pi).reduce(_+_))
    def rprop(x: (F[Double], F[Double])): (F[Double], F[Double]) =
      val (q, p) = x
      leapf(q, p)
    val mhk = mhKern(alpi, rprop)
    (q: F[Double]) =>
      val p = q map (qi => Gaussian(0, 1.0).draw())
      mhk((q, p))._1






object BridgeApp extends IOApp.Simple:

  val m = 100
  val leftFP = 1.0
  val rightFP = 4.0
  val b = Bridge(leftFP, rightFP, m)
  def mu(x: Double): Double = 1.1*x
  def dmu(x: Double): Double = 1.1
  def sig(x: Double): Double = 0.1*x
  def dsig(x: Double): Double = 0.1

  val dt = 1.0/m
  val sdt = math.sqrt(dt)
  def lpi(b: Bridge[Double]): Double =
    val x = leftFP +: b.br :+ rightFP
    (1 until x.length).toVector.map(i =>
      val xm = x(i-1)
      val xi = x(i)
      Gaussian(xm + mu(xm)*dt, sig(xm)*sdt).logPdf(xi)).reduce(_+_)
  def glpi(b: Bridge[Double]): Bridge[Double] =
    val br = (0 until m-1).toVector.map(i =>
      val xm = if (i == 0) leftFP else b.br(i-1)
      val x = b.br(i)
      val xp = if (i == m-2) rightFP else b.br(i+1)
      (xm + mu(xm)*dt - x)/(sig(xm)*sig(xm)*dt) -
        dsig(x)/sig(x) +
        (xm - x - mu(x)*dt)*(1 + dmu(x)*dt)/(sig(x)*sig(x)*dt) +
        math.pow(xm - x - mu(x)*dt, 2)*dsig(x)/(pow(sig(x), 3)*dt))
    Bridge(b.c, br)
  val kern = Kernels.hmcKernel(lpi, glpi, 0.0001, 20) // HMC tuning params
  val mcmc = LazyList.iterate(b)(kern).drop(10000).thin(100).take(100) // MCMC params

  def run = IO{ mcmc.foreach(println) }

