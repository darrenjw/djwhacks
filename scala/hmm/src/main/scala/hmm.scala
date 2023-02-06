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

  def forward[A](la: List[A], pi0: DVD)(f: A => DVD)(P: DMD) : List[DVD] =
    la.scanLeft(pi0)(forwardStep(P, f))


  def run = for
    x <- readData
    _ <- IO.println(x)
    fil = forward(x, DenseVector(0.5, 0.5))(
      (x: Double) => DenseVector(Gaussian(0.0, 1.0).pdf(x), Gaussian(3.0, 1.0).pdf(x)))(
      DenseMatrix((0.9, 0.1), (0.1, 0.9)))
    _ <- IO.println(fil)
  yield ExitCode.Success
