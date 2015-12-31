/*
errorhandling.scala

Handling errors with scalaz \/
Less powerful than using Validation

 */

// Error handling
//
// There are two types of errors:
// 1. Ones we can do something about (finite)
// 2. Ones we can't do anything about (infinite)

// Criteria / Goals:
// 1. We should handle all the errors we expect to handle
//
// 2. The compiler should shout at us if we add new errors we expect to handle
// 3. Fail-fast---when an error occurs we stop immediately

import scalaz.\/
import scalaz.syntax.either._

final case class User(id: Int)

object ApplicationCode {
  sealed trait HttpResponse
  final case class Ok(body: String) extends HttpResponse
  final case class NotFound(reason: String) extends HttpResponse

  import Database._

  def login(userId: Int): HttpResponse =
    Database.get(userId).fold(
      l = dbError => {
      dbError match {
        case UserNotFound => NotFound(s"Could not find $userId")
        case NotPermitted => NotFound(s"You shouldn't be here")
      }
    },
      r = user => Ok(s"Got you user ${user.id}")
    )

}

object Database {
  sealed trait DatabaseError
  final case object UserNotFound extends DatabaseError
  final case object NotPermitted extends DatabaseError

  def get(userId: Int): DatabaseError \/ User =
    userId match {
      case -1 => throw new Exception("Out of disk space!")
      case 0 => NotPermitted.left
      case 1 => UserNotFound.left
      case x => User(x).right
    }
}

object ErrorHandling {

  def main(args: Array[String]): Unit = {
    println(Database.get(0))
    println(Database.get(3))
  }

}
