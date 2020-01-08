/*
monix-test.scala

Trivial test from the monix gitter channel

*/

object MonixTest {

  import monix.reactive._
  import monix.execution.Scheduler.Implicits.global
  import scala.concurrent.duration._

  def main(args: Array[String]): Unit = {
    //val stream = Observable(1,2,3,4,5)
    //val stream = Observable.interval(1.second).take(10)
    val stream = Observable.intervalAtFixedRate(1.second).take(10).map(_.toInt)
    //val stream = Observable.fromIterable(1 to 10)
    stream.
      scan(0)(_+_).
      consumeWith(Consumer.foreach(a => println(a))).
      runSyncUnsafe(15.seconds)
  }

}

// eof

