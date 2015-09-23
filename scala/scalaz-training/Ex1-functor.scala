object Exercise1 {

 import scalaz.Functor
 import scalaz.syntax.functor._
 import scalaz.std.string._
 import scalaz.std.anyVal._
 import scalaz.std.option._

 def main(args: Array[String]): Unit = {

  sealed trait Result[+A]
  final case class Success[A](value: A) extends Result[A]
  final case class Warning[A](value: A, message: String) extends Result[A]
  final case class Failure(message: String) extends Result[Nothing]

  implicit object resultInstance extends Functor[Result] {
   def map[A,B](r: Result[A])(f: A => B): Result[B] = r match {
     case Success(r) => Success(f(r))
     case Warning(r,value) => Warning(f(r),value)
     case Failure(m) => Failure(m)
   }
  }

  val r = (Success(1): Result[Int]).map{_*2}

  println(r)

 }

}
