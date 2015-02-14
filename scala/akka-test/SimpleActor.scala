import akka.actor.Actor
import akka.actor.Actor.Receive

class SimpleActor extends Actor {
  override def receive: Receive = {
    case x =>
      println("Received message: " + x)
  }
}

