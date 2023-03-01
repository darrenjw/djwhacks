/*
tsqr.scala

Try implementing a TSQR

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import breeze.linalg.*
import breeze.numerics.*
import dev.ludovic.netlib.blas.BLAS.{ getInstance => blas }

import scala.concurrent.{ Future, Await, ExecutionContext }
import scala.concurrent.duration.*
import ExecutionContext.Implicits.global


type DMD = DenseMatrix[Double]


object Tsqr:

  def forwardSolve(A: DMD, Y: DMD): DMD =
    val yc = Y.copy
    blas.dtrsm("L", "L", "N", "N", yc.rows, yc.cols, 1.0, A.toArray,
      A.rows, yc.data, yc.rows)
    yc

  @annotation.tailrec
  def tsqR(x: DMD, blockSize: Int = 2500, threshold: Int = 10000): DMD =
    val n = x.rows
    if (n < threshold) qr.justR(x) else
      val blocks = if (n % blockSize == 0) (n / blockSize) else (n / blockSize + 1)
      val rMatsF = (0 until blocks).toVector.
        map(i => Future(i)).
        map(fi => fi.map(i => x((i*blockSize) until math.min(n, (i+1)*blockSize), ::))).
        map(fm => fm.map(m => qr.justR(m))).sequence
      val newXF = rMatsF.map(rMats => rMats.reduce(DenseMatrix.vertcat(_, _)))
      val newX = Await.result(newXF, 10.seconds)
      tsqR(newX, blockSize, threshold)

  def tsQr(x: DMD, r: DMD, blockSize: Int = 20000, threshold: Int = 30000): DMD =
    val n = x.rows  
    if (n < threshold) forwardSolve(r.t, x.t).t else
      val blocks = if (n % blockSize == 0) (n / blockSize) else (n / blockSize + 1)
      val qMatsF = (0 until blocks).toVector.
        map(i => Future(i)).
        map(fi => fi.map(i => x(i*blockSize until math.min(n, (i+1)*blockSize), ::))).
        map(fm => fm.map(m => forwardSolve(r.t, m.t).t)).sequence
      val qF = qMatsF.map(qMats => qMats.reduce(DenseMatrix.vertcat(_, _)))
      Await.result(qF, 10.seconds)

  def tsQR(x: DMD): (DMD, DMD) =
    val r = tsqR(x)
    (tsQr(x, r), r)

  def matClose(a: DMD, b: DMD, tol: Double = 1.0e-3): Boolean =
    val diff = a - b
    (sum(diff *:* diff) < tol)

  def time[A](f: => A): A =
    val s = System.nanoTime
    val ret = f
    println("time: "+(System.nanoTime - s)/1e6+"ms")
    ret


object TsqrApp extends IOApp.Simple:

  import Tsqr.*

  val m = DenseMatrix.rand(100000, 50)

  val r1 = time { qr.justR(m) }
 
  val r2 = time { tsqR(m) }

  val r3 = time { qr.justR(m) }

  val q1 = time { forwardSolve(r2.t, m.t).t }

  val q2 = time { tsQr(m, r2) }

  val q3 = time { forwardSolve(r2.t, m.t).t }

  val qr1 = time { qr.reduced(m) }
 
  val qr2 = time { tsQR(m) }


  def run =
    for
      _ <- IO.println( r2.rows )
      _ <- IO.println( matClose(abs(r1), abs(r2)) )
      _ <- IO.println( matClose(qr2._1 * qr2._2, m) )
      _ <- IO.println( matClose(q2.t * q2, DenseMatrix.eye[Double](q2.cols)) )
    yield ()
