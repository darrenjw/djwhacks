import scalaz.Validation
import scalaz.syntax.validation._
import scalaz.syntax.applicative._
import scalaz.std.list._
import scalaz.std.string._ 

object FormParser {
  val form = Map("name" -> "Noel", "age" -> "21")
  final case class User(name: String, age: Int)

  type Result[A] = Validation[List[String],A]

  def readName(form: Map[String, String]): Result[String] =
    form.get("name").fold(List("No name given").failure[String]){ name => name.success }

  def readAge(form: Map[String, String]): Result[Int] =
    form.get("age").fold(List("No age given").failure[Int]){ age =>
      parseInt(age).fold(
        fail = exn => List(s"$age is not an integer").failure,
        succ = age => age.success
      )
    }

  def readUser(form: Map[String, String]): Result[User] =
    (readName(form) |@| readAge(form)){ (name, age) => User(name, age) }
}
