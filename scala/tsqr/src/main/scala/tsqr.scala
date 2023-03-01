/*
tsqr.scala

Try implementing a TSQR

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import breeze.linalg.*
import breeze.numerics.*

type DMD = DenseMatrix[Double]


object TsqrApp extends IOApp.Simple:


  def matClose(a: DMD, b: DMD, tol: Double = 1.0e-3): Boolean =
    val diff = a - b
    (sum(diff *:* diff) < tol)

  def tsqr(x: DMD, blockSize: Int = 2000, threshold: Int = 5000): DMD =
    val n = x.rows
    val m = x.cols
    if (n < threshold) qr.justR(x) else
      val blocks = if (n % blockSize == 0) (n / blockSize) else (n / blockSize + 1)
      val rMats = (0 until blocks).toVector.
        map(i => x((i*blockSize) until math.min(n, (i+1)*blockSize), ::)).
        map(m => qr.justR(m))
      val newX = rMats.reduce(DenseMatrix.vertcat(_, _))
      tsqr(newX, blockSize, threshold)

  val m = DenseMatrix.rand(10000, 50)

  val r1 = qr.justR(m)
 
  val r2 = tsqr(m)

  def run =
    for
      _ <- IO.println( r2.rows )
      _ <- IO.println( matClose(abs(r1), abs(r2)) )
    yield ()
