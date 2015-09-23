object Exercise3 {

 import scalaz.Monad
 import scalaz.syntax.monad._
 import scalaz.std.string._
 import scalaz.std.anyVal._
 import scalaz.std.option._

 def main(args: Array[String]): Unit = {

  sealed trait Result[+A]
  final case class Success[A](value: A) extends Result[A]
  final case class Warning[A](value: A, message: String) extends Result[A]
  final case class Failure(message: String) extends Result[Nothing]

  implicit object resultInstance extends Monad[Result] {
   def point[A](a: => A): Result[A] = Success(a)
   def bind[A, B](fa: Result[A])(f: A => Result[B]): Result[B] = fa match {
     case Success(r) => f(r)
     case Warning(r,value) => f(r) match {
       case Success(r2) => Warning(r2,value)
       case Warning(r2,valu) => Warning(r2,value+" AND "+valu)
       case Failure(m) => Failure(m)
       }
     case Failure(m) => Failure(m)
   }
  }

  val r = (Success(1): Result[Int]).map{_*2}

  println(r)

  // better to use a non-empty list of Strings, or better still, a non-empty list of E
  // NonEmptyList - also need a semi group... and need to type alias over the E type so that
  // the monad only has one slot

 }

}
