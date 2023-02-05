/*
hmm.scala

HMM implementations

*/

import cats.*
import cats.implicits.*
import cats.effect.{IOApp, IO, ExitCode}

import breeze.linalg.*
import breeze.numerics.*

object HmmApp extends IOApp.Simple:

  def readData : IO[List[Double]] = IO {
    scala.io.Source.fromFile("short.txt").getLines.toList.map(_.toDouble)
  }


  def forwardStep[A](x: A, pi: DenseVector[Double], P: DenseMatrix[Double],
      f: (Int, A) => Double): DenseVector[Double] = ???

  def forward[A](x: List[A], pi0: DenseVector[Double])(
      f: (Int, A) => Double)(P: DenseMatrix[Double]) : List[DenseVector[Double]] = ???




  def run = for
    x <- readData
    _ <- IO.println(x)
  yield ExitCode.Success
