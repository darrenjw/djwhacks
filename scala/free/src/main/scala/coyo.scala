/*
coyo.scala

CoYoneda trick...

https://blog.oyanglul.us/grokking-monad/scala/en/part3
https://medium.com/@olxc/yoneda-and-coyoneda-trick-f5a0321aeba4
http://blog.higher-order.com/blog/2013/11/01/free-and-yoneda/

The Coyoneda is really just a free functor...

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

case class Coyo[F[_], A, B](fa: F[A], f: A => B):

  def map[C](fbc: B => C): Coyo[F, A, C] =
    Coyo(fa, (fbc compose f))

  def fold(fun: Functor[F]): F[B] = fun.map(fa)(f)

def lift[F[_], A](fa: F[A]): Coyo[F, A, A] = Coyo(fa, identity)


// a parameterised type, with no functor instance
case class Mytype[A](hidden: A)

// lift and then map at will
val x = lift(Mytype("hello")).map(_.length).map(_ * 2).map(_.toDouble)

// provide a functor instance to get a value back
val funct = new Functor[Mytype]:
  def map[A, B](fa: Mytype[A])(f: A => B): Mytype[B] = fa match
    case Mytype(x) => Mytype(f(x))

val y = x.fold(funct)


// Now using Coyoneda from Cats...
import cats.free.*
val z = Coyoneda.lift(Mytype("hello")).map(_.length).map(_ * 2)
val zz = z.run(funct) // use run() if given functor instance in scope


object CoYoApp extends IOApp.Simple:

  def display(s: String) = IO { println(s) }

  def run = for
    _ <- display("Hello")
    _ <- display(x.toString)
    _ <- display(y.toString)
    _ <- display(z.toString)
    _ <- display(zz.toString)
    _ <- display("Goodbye")
  yield ()

