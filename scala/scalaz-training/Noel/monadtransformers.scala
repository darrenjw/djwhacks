import scalaz.{OptionT,EitherT}
import scalaz.concurrent.Task
import scalaz.syntax.either._ // for .right
import scalaz.syntax.monad._

object MonadTransformers {
  def example1() = {
    type Error = String
    type Result[A] = EitherT[Task, Error, A]

    val result1 = EitherT(Task.now(1.right))
    val result2 = 1.point[Result]
    val result3: Result[Int] = EitherT.right(Task.now(1))

    val failed1 = EitherT(Task.now("Failed".left))
    val failed2: Result[Int] = EitherT.left(Task.now("Failed"))

    println(result1.run.run)
    println(result2.run.run)
  }

  def example2() = {
    type Error = String
    type Step2[A] = EitherT[Task, Error, A]
    type Result[A] = OptionT[Step2, A]

    val result1 = 1.point[Result]
    result1
  }
}
