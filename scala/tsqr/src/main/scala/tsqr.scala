/*
tsqr.scala

Try implementing a TSQR

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import breeze.linalg.*
import breeze.numerics.*

import scala.concurrent.{ Future, Await, ExecutionContext }
import scala.concurrent.duration.*
import ExecutionContext.Implicits.global


type DMD = DenseMatrix[Double]


object TsqrApp extends IOApp.Simple:


  def matClose(a: DMD, b: DMD, tol: Double = 1.0e-3): Boolean =
    val diff = a - b
    (sum(diff *:* diff) < tol)

  @annotation.tailrec
  def tsqr(x: DMD, blockSize: Int = 2000, threshold: Int = 5000): DMD =
    val n = x.rows
    val m = x.cols
    if (n < threshold) qr.justR(x) else
      val blocks = if (n % blockSize == 0) (n / blockSize) else (n / blockSize + 1)
      val rMatsF = (0 until blocks).toVector.
        map(i => Future(i)).
        map(fi => fi.map(i => x((i*blockSize) until math.min(n, (i+1)*blockSize), ::))).
        map(fm => fm.map(m => qr.justR(m))).sequence
      val newXF = rMatsF.map(rMats => rMats.reduce(DenseMatrix.vertcat(_, _)))
      val newX = Await.result(newXF, 10.seconds)
      tsqr(newX, blockSize, threshold)

  def time[A](f: => A): A =
    val s = System.nanoTime
    val ret = f
    println("time: "+(System.nanoTime - s)/1e6+"ms")
    ret

  val m = DenseMatrix.rand(10000, 20)

  val r1 = time { qr.justR(m) }
 
  val r2 = time { tsqr(m) }

  def run =
    for
      _ <- IO.println( r2.rows )
      _ <- IO.println( matClose(abs(r1), abs(r2)) )
    yield ()
