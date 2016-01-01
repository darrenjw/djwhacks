/*
parsercombinators.scala

Building an expression parser with free applicatives

 */

import scalaz.{Id, Applicative, ReaderWriterState, Monoid, FreeAp, ~>}
import scalaz.syntax.applicative._
import scalaz.syntax.monoid._
import scalaz.syntax.foldable._
import scalaz.std.list._
import scalaz.std.string._
import scalaz.std.anyVal._

object Parser {
  sealed trait Parser[A] {
    type Result = A

    def +(implicit m: Monoid[A]): FreeAp[Parser, A] =
      (Parser.lift(this) |@| Parser.star(this)) { (first, rest) => (first +: rest).foldMap() }
  }
  object Parser {
    def oneOf(choices: List[String]): FreeAp[Parser, String] =
      lift(OneOf(choices))

    def star[A](parser: Parser[A]): FreeAp[Parser, List[A]] =
      lift(Star(parser))

    def lift[A](parser: Parser[A]): FreeAp[Parser, A] =
      FreeAp.lift(parser)
  }
  final case class OneOf(choices: List[String]) extends Parser[String]
  final case class Star[A](parser: Parser[A]) extends Parser[List[A]]

  //1+1

  type ParserMachine[A] = ReaderWriterState[String, Unit, Int, A]
  object Interpreter extends (Parser ~> ParserMachine) {
    def apply[A](p: Parser[A]): ParserMachine[A] =
      p match {
        case Star(parser) =>
          ReaderWriterState[String, Unit, Int, A] {
            (input, offset) =>
              {
                var theOffset = offset
                var theResult = List.empty[parser.Result]
                val machine = apply(parser)

                try {
                  while (true) {
                    val (_, result, state) = machine.run(input, theOffset)
                    theResult = theResult :+ result
                    theOffset = state
                  }
                } catch {
                  case exn: Exception => ()
                }

                if (theOffset == offset)
                  throw new Exception("Didn't progress")
                else
                  ((), theResult, theOffset)
              }
          }
        case OneOf(choices) =>
          ReaderWriterState[String, Unit, Int, A] {
            (input: String, offset: Int) =>
              val head = input(offset).toString
              if (choices.contains(head))
                ((), head, offset + 1)
              else
                throw new Exception("AAAaarrrgggghh")
          }
      }
  }

  object Example {
    import Parser._

    val number = oneOf((0 to 9).map(_.toString).toList)
    val operator = oneOf(List("+", "-", "*", "/"))

    val numbers = star(OneOf((0 to 9).map(_.toString).toList))

    case class Expr(l: String, op: String, r: String)

    val expression = (numbers |@| operator |@| numbers) { (numbers1, op, numbers2) =>
      Expr.apply(numbers1.mkString, op, numbers2.mkString)
    }

    def run() = {
      expression.foldMap(Interpreter).eval("456+123", 0)
    }

    def error() = {
      expression.foldMap(Interpreter).eval("1@1", 0)
    }
  }

  def main(args: Array[String]): Unit = {
    println(Example.run())
    //println(Example.error())
  }

}
