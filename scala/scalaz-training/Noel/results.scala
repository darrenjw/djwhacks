/*
results.scala

Result type - result, errors and warnings, by
defining a Result Monad for use with scalaz

Defined here for an arbitrary Error type, E, which is a bit tricky, due to having two type parameters - need type projections...

 */

import scalaz.{Semigroup, Monad}
import scalaz.syntax.monad._ // for flatMap, map, etc.
import scalaz.syntax.semigroup._ // for |+|

sealed trait Result[+E, +A]

object Result {

  implicit def resultInstance[E](implicit s: Semigroup[E]): Monad[({ type l[A] = Result[E, A] })#l] = {
    type MonadResult[A] = Result[E, A]

    new Monad[MonadResult] {
      override def bind[A, B](fa: Result[E, A])(f: A => Result[E, B]): Result[E, B] =
        fa match {
          case Success(v) => f(v)
          case Warning(v, e) =>
            f(v) match {
              case Success(v2) => Warning(v2, e)
              case Warning(v2, e2) => Warning(v2, e |+| e2)
              case Failure(e2) => Failure(e |+| e2)
            }
          case Failure(e) => Failure(e)
        }

      override def point[A](a: => A): Result[E, A] =
        Success(a)
    }
  }

}

final case class Success[A](value: A) extends Result[Nothing, A]
final case class Warning[E, A](value: A, message: E) extends Result[E, A]
final case class Failure[E](message: E) extends Result[E, Nothing]

object ResultExample {

  import scalaz.std.string._
  import Result._

  def go() = {
    type ResultM[A] = Result[String, A]
    implicit val monadInstance: Monad[ResultM] = Result.resultInstance[String]
    println(5.point[ResultM])
    println(Monad[ResultM].point(5))
    println(3.point[ResultM].map(_ + 2))
    println(3.point[ResultM].flatMap(_ => Failure("Ooops!")))
  }

  def main(args: Array[String]): Unit = {
    go()
  }

}
