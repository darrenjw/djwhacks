/*
expr.scala

2*3 + 4*5 = 26

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}


object ClassicOO:

  object Calc:
    def literal(a: Int): Int = a
    def add(a: Int, b: Int): Int = a + b
    def mult(a: Int, b: Int): Int = a * b

  import Calc.*
  val result = add(mult(literal(2), literal(3)), mult(literal(4), literal(5)))

object ClassicOOPretty:

  object Pret:
    def literal(a: Int): String = a.toString
    def add(a: String, b: String): String = "(" + a + " + " + b + ")"
    def mult(a: String, b: String): String = "(" + a + " * " + b + ")"

  import Pret.*
  val result = add(mult(literal(2), literal(3)), mult(literal(4), literal(5)))


object ClassicFP:

  enum Calc:
    case Literal(a: Int)
    case Add(a: Calc, b: Calc)
    case Mult(a: Calc, b: Calc)

  import Calc.*

  val prog = Add(Mult(Literal(2), Literal(3)), Mult(Literal(4), Literal(5)))

  def eval(c: Calc): Int = c match
    case Literal(a) => a
    case Add(a, b) => eval(a) + eval(b)
    case Mult(a, b) => eval(a) * eval(b)

  val result = eval(prog)

  def pret(c: Calc): String = c match
    case Literal(a) => a.toString
    case Add(a, b) => "(" + pret(a) + " + " + pret(b) + ")"
    case Mult(a, b) => "(" + pret(a) + " * " + pret(b) + ")"

  val pretty = pret(prog)


object GenericFP:

  enum Calc[A]:
    case Literal(a: A)
    case Add(a: Calc[A], b: Calc[A])
    case Mult(a: Calc[A], b: Calc[A])

  import Calc.*

  val prog = Add(Mult(Literal(2), Literal(3)), Mult(Literal(4), Literal(5)))

  val dprog = Add(Mult(Literal(2.0), Literal(3.0)), Mult(Literal(4.0), Literal(5.0)))

  import math.Numeric.Implicits.infixNumericOps

  def eval[A: Numeric](c: Calc[A]): A = c match
    case Literal(a) => a
    case Add(a, b) => eval(a) + eval(b)
    case Mult(a, b) => eval(a) * eval(b)

  val result = eval(prog)

  val dresult = eval(dprog)


object ObjectAlgebra:

  sealed trait Calc[A]:
    def literal(a: Int): A // Note that "a" needs to be an Int (A is output type not input type)
    def add(a: A, b: A): A
    def mult(a: A, b: A): A

  //def expr[A](c: Calc[A]): A =
  //val expr = [A] => (c: Calc[A]) =>
  val expr: [A] => Calc[A] => A = [A] => (c: Calc[A]) =>
    import c.*
    add(mult(literal(2), literal(3)), mult(literal(4), literal(5)))
  // "expr" is a polymorphic function - not a regular "value" - can't introspect, but can pass around

  object Eval extends Calc[Int]:
    def literal(a: Int): Int = a    
    def add(a: Int, b: Int): Int = a + b
    def mult(a: Int, b: Int): Int = a * b

  val result = expr(Eval)

  object Pret extends Calc[String]:
    def literal(a: Int): String = a.toString
    def add(a: String, b: String): String = "(" + a + " + " + b + ")"
    def mult(a: String, b: String): String = "(" + a + " * " + b + ")"

  val pretty = expr(Pret)


object MonadicFP:

  enum Calc:
    case Literal(a: Int)
    case Add(a: Calc, b: Calc)
    case Mult(a: Calc, b: Calc)

  import Calc.*

  val prog = Add(Mult(Literal(2), Literal(3)), Mult(Literal(4), Literal(5)))

  def eval[M[_]: Monad](c: Calc): M[Int] = c match
    case Literal(a) => Monad[M].pure(a)
    case Add(ea, eb) =>
      for
        a <- eval(ea)
        b <- eval(eb)
      yield (a + b)
    case Mult(ea, eb) =>
      for
        a <- eval(ea)
        b <- eval(eb)
      yield (a * b)

  val result = eval[Option](prog)

  val delayed = eval[cats.Eval](prog)


object ExprApp extends IOApp.Simple:

  def display(s: String) = IO { println(s) }

  def run = for
    _ <- display(ClassicOO.result.toString)
    _ <- display(ClassicOOPretty.result)
    _ <- display(ClassicFP.prog.toString)
    _ <- display(ClassicFP.result.toString)
    _ <- display(ClassicFP.pretty)
    _ <- display(GenericFP.result.toString)
    _ <- display(GenericFP.dresult.toString)
    _ <- display(ObjectAlgebra.expr.toString)
    _ <- display(ObjectAlgebra.result.toString)
    _ <- display(ObjectAlgebra.pretty)
    _ <- display(MonadicFP.result.toString)
    _ <- display(MonadicFP.delayed.value.toString)
  yield ()

