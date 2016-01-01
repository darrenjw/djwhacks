/*
freemonads-dave.scala

Dave Gurnell's version of the free monad expression parser examples

This code extends App, so doesn't need a main method


 */

import scalaz._
import scalaz.Scalaz._

object FreeExample extends App {

  sealed trait Expr[A]
  final case class Add(a: Int, b: Int) extends Expr[Int]
  final case class Mul(a: Int, b: Int) extends Expr[Int]
  final case class Lit(a: Int) extends Expr[Int]

  type Expressable[A] = Coyoneda[Expr, A]

  object Expr {
    def add(a: Int, b: Int): Free[Expressable, Int] = Free.liftFC(Add(a, b))
    def mul(a: Int, b: Int): Free[Expressable, Int] = Free.liftFC(Mul(a, b))
    def lit(a: Int): Free[Expressable, Int] = Free.liftFC(Lit(a))
  }

  object IdInterpreter extends (Expr ~> Id.Id) {
    def apply[A](expr: Expr[A]): Id.Id[A] = expr match {
      case Add(a, b) => a + b
      case Mul(a, b) => a * b
      case Lit(a) => a
    }
  }

  type Log[A] = Writer[List[String], A]
  object LogInterpreter extends (Expr ~> Log) {
    def apply[A](expr: Expr[A]): Log[A] = expr match {
      case msg @ Add(a, b) => List(s"$msg").tell as (a + b)
      case msg @ Mul(a, b) => List(s"$msg").tell as (a * b)
      case msg @ Lit(a) => List(s"$msg").tell as (a)
    }
  }

  type Count[A] = State[Int, A]
  object CountInterpreter extends (Id.Id ~> Count) {
    def apply[A](value: Id[A]): Count[A] =
      value.state leftMap (_ + 1)
  }

  val example = {
    import Expr._
    for {
      a <- lit(1)
      b <- lit(2)
      x <- mul(a, b)
      c <- lit(3)
      d <- lit(4)
      y <- mul(c, d)
      sum <- add(x, y)
    } yield sum
  }

  println(Free.runFC(example)(LogInterpreter).run)
  println(Free.runFC(example)(CountInterpreter compose IdInterpreter).run(0))
}
