object Exercise7 {

 import scalaz.{Free, ~>, Id, Coyoneda}

 def main(args: Array[String]): Unit = {

  sealed trait Parser[A]
  final case class OneOf(choices: Seq[String]) extends Parser[A]
  
  object Example {
    val number = OneOf((0 to 9).map{_.toString}).+
    val operator = OneOf("+","-","*","/")
    val expression = number |@| operator |@| number

   println("hi")
 }

}
