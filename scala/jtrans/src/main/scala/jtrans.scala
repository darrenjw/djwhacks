/*
jtrans.scala
Stub for Scala Cats code
*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}
import org.jtransforms.dct.*

def printLn(s: String): IO[Unit] = IO{println(s)}

// DCT using JTransforms (1d)
def dct(v: Vector[Double], inverse: Boolean = false): Vector[Double] =
  val o = DoubleDCT_1D(v.length)
  val c = v.toArray
  if (inverse)
    o.inverse(c, false)
  else
    o.forward(c, false)
  if (inverse)
    c.toVector.map(_ * v.length)
  else
    c.toVector.map(_ / v.length)

def dct2(m: Vector[Vector[Double]], inverse: Boolean = false): Vector[Vector[Double]] =
  val N = m.length * m(0).length
  val o = DoubleDCT_2D(m.length, m(0).length)
  val c = m.map(_.toArray).toArray
  if (inverse)
    o.inverse(c, false)
  else
    o.forward(c, false)
  if (inverse)
    c.map(_.toVector).map(r => r.map(_ * (N.toDouble/2.0))).toVector
  else
    c.map(_.toVector).map(r => r.map(_ * (2.0/N.toDouble))).toVector

// 2d inverse DCT just seems wrong...


// Some helper functions
extension (v: Vector[Double]) def *:*(w: Vector[Double]): Vector[Double] =
    (v zip w) map (_ * _)

def norm(v: Vector[Double]): Double = math.sqrt((v *:* v).sum)

def trans(ma: Vector[Vector[Double]]): Vector[Vector[Double]] =
  val rows = ma(0).length
  (0 until rows).toVector map (i => ma map (cj => cj(i)))

// Naive DCT functions, for verification
def dct0(x: Vector[Double]): Vector[Double] =
    val N = x.length
    val cf = Vector.tabulate(N){n => math.Pi*(n + 0.5)/N}
    val X = Vector.tabulate(N){k => (x *:* (cf.map(_ * k.toDouble)).map(math.cos(_))).sum}
    X.map(_ * (2.0 / N))

def idct0(x: Vector[Double]): Vector[Double] =
    val N = x.length
    val cf = Vector.tabulate(N-1){k => math.Pi*(k+1)/N}
    Vector.tabulate(N){n => x(0)/2 + (x.tail *:* (cf.map(_ * (n+0.5)).map(math.cos(_)))).sum}

def dct20(m: Vector[Vector[Double]]): Vector[Vector[Double]] =
  trans(trans(m.map(dct0(_))).map(dct0(_)))

def idct20(m: Vector[Vector[Double]]): Vector[Vector[Double]] =
  trans(trans(m.map(idct0(_))).map(idct0(_)))



val v: Vector[Double] = Vector(1,2,2,1,2,2,2,3,3,4,3,3,3,4,3,2,2,1,0)
val t0 = dct0(v)
val vv0 = idct0(t0)
val t = dct(v)
val vv = dct(t, true)

val m: Vector[Vector[Double]] = Vector(Vector(1,2,3), Vector(2,4,1))
val mt0 = dct20(m)
val mm0 = idct20(mt0)
val mt = dct2(m)
val mm = dct2(mt, true)

def mainIO: IO[Unit] = for
  _ <- printLn("Hello")
  _ <- printLn(v.toString)
  _ <- printLn(t.toString)
  _ <- printLn(t0.toString)
  _ <- printLn(vv.toString)
  _ <- printLn(vv0.toString)
  _ <- printLn(m.toString)
  _ <- printLn(trans(m).toString)
  _ <- printLn(mt.toString)
  _ <- printLn(mt0.toString)
  _ <- printLn(mm.toString)
  _ <- printLn(mm0.toString)
  _ <- printLn("Goodbye")
yield ()

object CatsApp extends IOApp.Simple:
  def run = mainIO

