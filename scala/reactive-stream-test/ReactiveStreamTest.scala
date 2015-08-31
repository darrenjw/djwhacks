// My reactive stream test

import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import akka.stream.scaladsl.{Source, Sink, Flow}

import java.lang.management._

object ReactivesStream {

  def main(args: Array[String]): Unit = {
    implicit val system = ActorSystem("Sys")
    import system.dispatcher
    implicit val materializer = ActorMaterializer()

    val osb=ManagementFactory.getOperatingSystemMXBean()

    val mySource = Source(() => Iterator.continually(osb.getSystemLoadAverage()))

    val myFlow = Flow[Double].
      map(_*100).
      map(_.toInt)

    val myThrottle = Flow[Int].
      map(x => {Thread.sleep(2000) ; x})

    val mySink = Sink.foreach(println)

    mySource.via(myFlow).via(myThrottle).runWith(mySink).
      onComplete(_ => system.shutdown())

  }
}

// eof


