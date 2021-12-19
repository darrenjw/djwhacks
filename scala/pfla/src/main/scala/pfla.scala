/*
pfla.scala

Parallel functional linear algebra in Scala 3

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

object Pfla:

  import scala.collection.parallel.immutable.ParVector
  import scala.collection.parallel.CollectionConverters.*

  // Vec is a thin layer over a ParVector
  type Vec[A] = ParVector[A]
  // Mat is a Vec of column Vecs (so column-major matrices)
  type Mat[A] = Vec[Vec[A]]

  // constructor in companion object
  object Vec:
    def apply[A](a: A*): Vec[A] = a.toVector.par

  // append two vectors, but will also column-bind two matrices
  extension [A] (va: Vec[A])
    def append(va2: Vec[A]): Vec[A] = va ++ va2

  // add two numeric vectors
  extension [A: Numeric] (va: Vec[A])
    def +(va2: Vec[A]): Vec[A] =
      (va zip va2).map(summon[Numeric[A]].plus(_,_))

  // hadamard (elementwise) product of two numeric vectors
  extension [A: Numeric] (va: Vec[A])
    def had(va2: Vec[A]): Vec[A] =
      (va zip va2).map(summon[Numeric[A]].times(_,_))

  // dot product of two numeric vectors
  extension [A: Numeric] (va: Vec[A])
    def dot(va2: Vec[A]): A =
      (va had va2).reduce(summon[Numeric[A]].plus(_,_))

  // vec of a matrix (stack the columns)
  extension [A] (vva: Mat[A])
    def vec: Vec[A] = vva.reduce(_ append _)

  // transpose a vector
  extension [A: Numeric] (va: Vec[A]) // Numeric constraint to prevent application to Mat
    def t: Mat[A] = va map (Vec(_))
  
  // transpose a matrix
  extension [A] (ma: Mat[A])
    def t: Mat[A] =
      val rows = ma(0).length
      (0 until rows).toVector.par map (i => ma map (cj => cj(i)))

  // matrix-vector multiply
  extension [A: Numeric] (ma: Mat[A])
    def times(va: Vec[A]): Vec[A] =
      ma.t map (_ dot va)

  // matrix-matrix multiply
  extension [A: Numeric] (ma: Mat[A])
    def *(ma2: Mat[A]): Mat[A] =
      val mat = ma.t
      ma2 map (c => mat map (_ dot c))

  // matrix addition
  extension [A: Numeric] (ma: Mat[A])
    def plus(ma2: Mat[A]): Mat[A] =
      (ma zip ma2) map (p => p._1 + p._2)




object PflaApp extends IOApp.Simple:

  import Pfla.*

  val l = Vec(1,2,3)
  val v = Vec(3,5)
  val ll = l append l
  val l4 = ll append ll
  val ls = l had l

  val m = Vec(Vec(1,2,3), Vec(4,5,6))
  val vm = m.vec

  def run = for
    _ <- IO{ println(l4) }
    _ <- IO{ println(m) }
    _ <- IO{ println(l dot l) }
    _ <- IO{ println(v + v) }
    _ <- IO{ println(l.t.t * l.t) }
    _ <- IO{ println(l.t.t.vec) }
    _ <- IO{ println(m times v) }
    _ <- IO{ println(m * m.t) }
    _ <- IO{ println(m.t * m) }
    _ <- IO{ println(m plus m) }
  yield ()
