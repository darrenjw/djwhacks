package sample.stream

import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import akka.stream.scaladsl.{ Source, Sink, Flow }

object MyStream {

  def main(args: Array[String]): Unit = {
    implicit val system = ActorSystem("Sys")
    import system.dispatcher
    implicit val materializer = ActorMaterializer()

    val text =
      """|Lorem Ipsum is simply dummy text of the printing and typesetting industry.
         |Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
         |when an unknown printer took a galley of type and scrambled it to make a type
         |specimen book.""".stripMargin

    val mySource = Source(() => text.split("\\s").iterator)

    val myFlow = Flow[String].
      map(_.toUpperCase).
      filter(_.length > 3)

    val mySink = Sink.foreach(println)

    mySource.via(myFlow).runWith(mySink).
      onComplete(_ => system.shutdown())

  }
}
