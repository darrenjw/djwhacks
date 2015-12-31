/*
implicits.scala

Typeclasses and implicits in scala (no scalaz in here)

Monoids and Functors



Type classes:

trait Monoid[A] {
  def append(a1: A, a2: A): A
  def zero: A
}
object Monoid {
  def apply[A](implicit monoid: Monoid[A]): Monoid[A] =
    monoid
}

object Instances {
  implicit object intInstance extends Monoid[Int] {
    def append(a1: Int, a2: Int): Int = a1 + a2
    def zero: Int = 0
  }
}

object Example {
  import Instances._ // Get implicit values into scope
  Monoid[Int].append(1, 2)
  Monoid.apply[Int].append(1, 2)
  Monoid.apply(intInstance).append(1, 2)
}

object Syntax {
  implicit class monoidOps[A](a: A) {
    def |+|(other: A)(implicit monoid: Monoid[A]): A =
      Monoid[A].append(a, other)
  }
}

 */

import scala.language.higherKinds

object FunctorExample {

  trait Functor[F[_]] {
    def map[A, B](fa: F[A])(f: A => B): F[B]
  }

  implicit object optionInstance extends Functor[Option] {
    def map[A, B](fa: Option[A])(f: A => B): Option[B] =
      fa.map(f)
  }

  implicit class FunctorOps[F[_], A](fa: F[A]) {
    def map[B](f: A => B)(implicit ev: Functor[F]): F[B] =
      ev.map(fa)(f)
  }

  def main(args: Array[String]): Unit = {
    println(Some(1) map {_*2})
}

}

