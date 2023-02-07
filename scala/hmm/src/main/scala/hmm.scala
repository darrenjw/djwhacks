/*
hmm.scala

HMM implementations

*/

import cats.*
import cats.implicits.*
import cats.effect.{IOApp, IO, ExitCode}

import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.Rand.VariableSeed.randBasis

import monocle.Lens

type DVD = DenseVector[Double]
type DMD = DenseMatrix[Double]

object HmmApp extends IOApp.Simple:

  // Some generic HMM functions

  def forwardStep[A](P: DMD, f: A => DVD)(pi: DVD, a: A): DVD =
    val unn = f(a) *:* (P.t * pi)
    unn / sum(unn)

  def forward[A](la: List[A], pi0: DVD)(f: A => DVD)(P: DMD): List[DVD] =
    la.scanLeft(pi0)(forwardStep(P, f))

  def backStep[A](P: DMD)(s: DVD, f: DVD): DVD =
    f *:* (P * (s /:/ (P.t * f))) // could explicitly normalise for stability if necessary

  def back[A](fpl: List[DVD])(P: DMD): List[DVD] =
    val rev = fpl.//drop(1). // don't drop one here to smooth back to time 0
      reverse
    val init = rev.head
    rev.drop(1).scanLeft(init)(backStep(P)).reverse

  def smooth[A](la: List[A], pi0: DVD)(f: A => DVD)(P: DMD): List[DVD] =
    back(forward(la, pi0)(f)(P))(P)


  // Show how to do forward filtering with a (parallel) prefix scan
  // illustrative - will underflow for big data sets currently - but easy to fix
  // eg. could have a combine operation that rescales after multiply

  def forwardPS[A](la: List[A], pi0: DVD)(f: A => DVD)(P: DMD): List[DVD] =
    val lfap = la map (a => P * diag(f(a)))
    val ps = lfap.scan(DenseMatrix.eye[Double](P.rows))(_ * _)
    val unn = ps map (_.t * pi0)
    val s = unn map (sum(_))
    (unn zip s) map ((vi, si) => vi / si)


  // Some lens based functions
  //  illustrative - not for big data sets - will blow stack

  def lens[A](la: List[A], P: DMD, f: A => DVD): Lens[DVD, DVD] =
    val ll = la.map(a => Lens(forwardStep(P, f)(_, a))((s: DVD) => backStep(P)(s, _)))
    // compose the list of lenses into a single lens and return the composed lens
    ll reduce (_ andThen _)


  // Set up example

  def readData : IO[List[Double]] = IO(
    scala.io.Source.fromFile("short.txt").getLines.toList.map(_.toDouble)
  )

  val pi0 = DenseVector(0.5, 0.5)

  val P = DenseMatrix((0.9, 0.1), (0.1, 0.9))

  def model(x: Double): DVD =
    DenseVector(Gaussian(0.0, 1.0).pdf(x), Gaussian(3.0, 1.0).pdf(x))


  def run = for
    x <- readData
    _ <- IO.println(x)
    smo = smooth(x, pi0)(model)(P)
    _ <- IO.println(smo)
    _ <- IO.println(smo.head)
    _ <- IO.println(smo.drop(1).head)
    l = lens(x, P, model)
    _ <- IO.println(l.get(pi0))
    res = l.replace(l.get(pi0))(pi0)
    _ <- IO.println(res)
    fo = forward(x, pi0)(model)(P)
    _ <- IO.println(fo)
    fops = forwardPS(x, pi0)(model)(P)
    _ <- IO.println(fops)
  yield ExitCode.Success
