object Exercise4 {

 import scalaz.EitherT
 import scalaz.concurrent.Task
 import scalaz.syntax.either._
 import scalaz.syntax.monad._

 def example1() = {

  type Error = String
  type Result[A] = EitherT[Task,Error,A]

  val result1 = EitherT(Task.now(1.right))
  val result2 = 1.point[Result]

  val failed1 = EitherT(Task.now("Failed".left))
  val failed2 = EitherT.left(Task.now("Failed"))

  println(result1.run.run)
  println(result2.run.run)
}

 def main(args: Array[String]): Unit = {

  example1()

  import scalaz.OptionT
  //type Result[A] = ??? // Future[Error \/ Option[A]]
  type Step2[A] = EitherT[Task,Error,A]
  type Result[A] = OptionT[Step2,A]
  val result1 = 1.point[Result]
  println(result1.run.run.run)

 }

}
