/*
trampolines.scala

Exploration of trampolining ideas

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


*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

def logfact1(n: Int): Double =
  if (n <= 1)
    0.0
  else
    math.log(n) + logfact1(n-1)

@annotation.tailrec
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
  @annotation.tailrec
  final def run: A = this match
    case Value(a) => a
    case Wrap(t) => t().run
  @annotation.tailrec
  final def flatMap[B](fab: A => Tramp[B]): Tramp[B] = this match
    case Value(a) => fab(a)
    case Wrap(t) => t().flatMap(fab)

case class Value[+A](a: A) extends Tramp[A]
case class Wrap[+A](t: () => Tramp[A]) extends Tramp[A]

def logfact8(n: Int): Tramp[Double] =
  if (n <=1)
    Value(0.0)
  else
    Wrap(() => logfact8(n-1).flatMap(a => Value(a + math.log(n))))




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
    _ <- IO{ println(logfact8(1000).run) }
  yield ()

