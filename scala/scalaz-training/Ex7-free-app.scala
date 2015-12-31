/*
Ex7-free-app.scala

Free applicatives

Didn't complete...

 */

object Exercise7 {

  import scalaz.{Free, ~>, Id, Coyoneda}

  sealed trait Parser[A]
  final case class OneOf(choices: Seq[String]) extends Parser[String]

  def main(args: Array[String]): Unit = {
    println("hi")
  }

  object Example {
    val number = OneOf((0 to 9).map { _.toString })
    val operator = OneOf(List("+", "-", "*", "/"))
    //val expression = number |@| operator |@| number
  }

}
