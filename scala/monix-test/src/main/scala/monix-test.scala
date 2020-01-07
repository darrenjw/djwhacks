/*
monix-test.scala
Stub for Scala Cats code
*/

object CatsApp {

  import cats._
  import cats.implicits._

  import monix.reactive._
  import monix.execution.Scheduler.Implicits.global
  import scala.concurrent.duration._

  def main(args: Array[String]): Unit = {
    Observable(1,2,3,4,5).
      consumeWith(Consumer.foreach(a => println(a))).
      runSyncUnsafe(5.seconds)
  }

}
