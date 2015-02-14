import akka.actor.{Props, ActorRef, ActorSystem}

object Main extends App {
  val system = ActorSystem("the-actor-system")
  val simpleActor: ActorRef = system.actorOf(Props[SimpleActor])
  simpleActor ! "Hello world"

  // Wait for message processing and shutdown
  // (not needed in normal application)
  java.lang.Thread.sleep(1000)
  system.shutdown()
}

