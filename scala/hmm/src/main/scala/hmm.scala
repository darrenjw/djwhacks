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

type DVD = DenseVector[Double]
type DMD = DenseMatrix[Double]

object HmmApp extends IOApp.Simple:

  def readData : IO[List[Double]] = IO(
    scala.io.Source.fromFile("short.txt").getLines.toList.map(_.toDouble)
  )

  def forwardStep[A](P: DMD, f: A => DVD)(pi: DVD, a: A): DVD =
    val unn = f(a) *:* (P.t * pi)
    unn / sum(unn)

  def forward[A](la: List[A], pi0: DVD)(f: A => DVD)(P: DMD): List[DVD] =
    la.scanLeft(pi0)(forwardStep(P, f))

  def backStep[A](P: DMD)(s: DVD, f: DVD): DVD =
    f *:* (P * (s /:/ (P.t * f)))

  def back[A](fpl: List[DVD])(P: DMD): List[DVD] =
    val rev = fpl.drop(1).reverse
    val init = rev.head
    rev.drop(1).scanLeft(init)(backStep(P)).reverse

  def smooth[A](la: List[A], pi0: DVD)(f: A => DVD)(P: DMD): List[DVD] =
    back(forward(la, pi0)(f)(P))(P)





  val pi0 = DenseVector(0.5, 0.5)
  val P = DenseMatrix((0.9, 0.1), (0.1, 0.9))

  def run = for
    x <- readData
    _ <- IO.println(x)
    smo = smooth(x, pi0)(
      (x: Double) => DenseVector(Gaussian(0.0, 1.0).pdf(x), Gaussian(3.0, 1.0).pdf(x)))(
      P)
    _ <- IO.println(smo)
  yield ExitCode.Success
