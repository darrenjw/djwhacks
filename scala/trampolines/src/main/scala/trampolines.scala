/*
trampolines.scala

Exploration of trampolining ideas (including tailRecM)

Cats docs:
 *  https://typelevel.org/cats/api/cats/Eval$.html
 *  https://typelevel.org/cats/typeclasses/monad.html
 *  https://typelevel.org/cats/faq.html#tailrecm

Key papers:
 *  https://blog.higher-order.com/assets/trampolines.pdf
 *  https://functorial.com/stack-safety-for-free/index.pdf

A few blog posts:
 *  https://edward-huang.com/functional-programming/programming/scala/optimization/2021/01/17/common-pattern-of-creating-stack-safe-recursion/
 *  https://medium.com/@alexander.zaidel/stack-safe-monads-33e803065a9d
 *  https://free.cofree.io/2017/08/24/trampoline/
 *  https://medium.com/@olxc/trampolining-and-stack-safety-in-scala-d8e86474ddfa



 Default implementation of tailRecM - typically not stack-safe:

  def tailRecM[A, B](a: A)(f: A => F[Either[A, B]]): F[B] =
    flatMap(f(a)) {
      case Right(b) => pure(b)
      case Left(nextA) => tailRecM(nextA)(f)
    }



*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}
import annotation.tailrec

def logfact1(n: Int): Double =
  if (n <= 1)
    0.0
  else
    math.log(n) + logfact1(n-1)

@tailrec
def logfact2(n: Int, acc: Double = 0.0): Double =
  if (n <= 1)
    acc
  else
    logfact2(n-1, math.log(n) + acc)

def nats = LazyList.iterate(1)(_+1)
def logfact3 = nats.scanLeft(0.0)(_ + math.log(_))

// works because foldLeft is stack safe, but natural structural recursion is more
// like a foldRight, which is not...

class Thunk[A](a: => A):
  def run = a
  // def flatMap[B](fab: A => Thunk[B]): Thunk[B] = fab(a)

def logfact4(n: Int): Thunk[Double] =
  if (n <=1)
    Thunk(0.0)
  else
    Thunk(math.log(n) + logfact4(n-1).run)

def logfact5(n: Int): Eval[Double] =
  if (n <=1)
    Eval.always(0.0)
  else
    Eval.always(math.log(n) + logfact5(n-1).value)

def logfact6(n: Int): Eval[Double] =
  if (n <=1)
    Eval.always(0.0)
  else
    logfact6(n-1).flatMap(a => Eval.always(a + math.log(n)))

def logfact7(n: Int): Eval[Double] =
  if (n <=1)
    Eval.always(0.0)
  else
    Eval.defer(logfact7(n-1).flatMap(a => Eval.always(a + math.log(n))))

sealed trait Tramp[+A]:
  @tailrec
  final def run: A = this match
    case Value(a) => a
    case Wrap(t) => t().run
    case FlatMap(ta, fab) => ta match
      case Value(a) => fab(a).run
      case Wrap(t) => t().flatMap(fab).run
      case FlatMap(tz, fza) => FlatMap(tz, z => FlatMap(fza(z), fab)).run
  final def flatMap[B](fab: A => Tramp[B]): Tramp[B] =
    FlatMap(this, fab)

case class Value[+A](a: A) extends Tramp[A]
case class Wrap[+A](t: () => Tramp[A]) extends Tramp[A]
case class FlatMap[A,B](a: Tramp[A], f: A => Tramp[B]) extends Tramp[B]

// for a trampolined monad, the default tailRecM implementation is stack-safe
object Tramp:
  final def tailRecM[A, B](a: A)(f: A => Tramp[Either[A, B]]): Tramp[B] =
    f(a).flatMap{
      case Right(b) => Value(b)
      case Left(nextA) => tailRecM(nextA)(f)
    }

def logfact8(n: Int): Tramp[Double] =
  if (n <=1)
    Value(0.0)
  else
    Wrap(() => logfact8(n-1).flatMap(a => Value(a + math.log(n))))

def logfact9(n: Int): Eval[Double] =
  Monad[Eval].tailRecM((n, 0.0)){ (n: Int, a: Double) =>
    if (n <= 1) Eval.always(Right(a))
    else Eval.later(Left((n-1, a + math.log(n))))
  }

sealed trait Trampoline[A]:
  @tailrec
  final def run: A = this match
    case Done(a) => a
    case More(th) => th().run
  @tailrec
  final def flatMap[B](fab: A => Trampoline[B]): Trampoline[B] = this match
    case Done(a) => fab(a)
    case More(th) => th().flatMap(fab)

// Can't write a safe tailRecM without the FlatMap trick? Probably can, but looks tricky!
// Can probably modify the instance for List or Stream (LazyList) in Cats
object Trampoline:
  final def tailRecM[A, B](a: A)(f: A => Trampoline[Either[A, B]]): Trampoline[B] =
    f(a).flatMap{
      case Right(b) => Done(b)
      case Left(nextA) => tailRecM(nextA)(f)
    }

case class Done[A](a: A) extends Trampoline[A]
case class More[A](th: () => Trampoline[A]) extends Trampoline[A]

def logfact10(n: Int): Trampoline[Double] =
  if (n <= 1) Done(0.0)
  else More(() => logfact10(n-1).flatMap(a => Done(a + math.log(n))))

def logfact11(n: Int): Trampoline[Double] =
  Trampoline.tailRecM((n, 0.0)){ (n: Int, a: Double) =>
    if (n <= 1) Done(Right(a))
    else More(() => Done(Left((n-1, a + math.log(n)))))
  }

def logfact12(n: Int): Tramp[Double] =
  Tramp.tailRecM((n, 0.0)){ (n: Int, a: Double) =>
    if (n <= 1) Value(Right(a))
    else Wrap(() => Value(Left((n-1, a + math.log(n)))))
  }

// tailRecM is monadic generalisation of "tailRec" abstraction for tail recursion
@tailrec
def tailRec[A, B](a: A)(f: A => Either[A, B]): B = f(a) match
  case Right(b) => b
  case Left(nextA) => tailRec(nextA)(f)

// version for a language without tail call elimination:
def tailRec2[A, B](a: A)(f: A => Either[A, B]): B =
  var currentA = a
  var result: Option[B] = None
  while(result == None)
    f(currentA) match
      case Right(b) => result = Some(b)
      case Left(nextA) => currentA = nextA
  result.get

def logfact13(n: Int): Double =
  tailRec((n, 0.0)){ (n: Int, a: Double) =>
    if (n <= 1) Right(a)
    else Left((n-1, a + math.log(n)))
  }

object CatsApp extends IOApp.Simple:
  def run = for
    _ <- IO{ println(logfact1(1000)) }
    _ <- IO{ println(logfact2(1000000)) }
    _ <- IO{ println(logfact3(1000000)) }
    _ <- IO{ println(logfact4(1000000)) }
    _ <- IO{ println(logfact4(1000).run) }
    _ <- IO{ println(logfact5(1000000)) }
    _ <- IO{ println(logfact5(1000).value) }
    _ <- IO{ println(logfact6(1000)) }
    _ <- IO{ println(logfact6(1000).value) }
    _ <- IO{ println(logfact7(1000000)) }
    _ <- IO{ println(logfact7(1000000).value) }
    _ <- IO{ println(logfact8(1000000)) }
    _ <- IO{ println(logfact8(1000000).run) }
    _ <- IO{ println(logfact9(1000000)) }
    _ <- IO{ println(logfact9(1000000).value) }
    _ <- IO{ println(logfact10(1000000)) }
    _ <- IO{ println(logfact10(1000).run) }
    _ <- IO{ println(logfact11(1000)) }
    _ <- IO{ println(logfact11(1000).run) }
    _ <- IO{ println(logfact12(1000000)) }
    _ <- IO{ println(logfact12(1000000).run) }
    _ <- IO{ println(logfact13(1000000)) }
  yield ()

