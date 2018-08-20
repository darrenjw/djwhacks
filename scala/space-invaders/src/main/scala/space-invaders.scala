/*
Stub.scala
Stub for Scala Breeze code
*/

object Stub {

  case class SState(x: Int, y: Int, speed: Int)

  val width = 35
  val height = 15
  val player = SState(width/2, height -1, 0)

  def updateShip(ship: SState): SState = {
    val proposedX = ship.x + ship.speed
    if (proposedX >= width) {
      SState(width - (proposedX - width + 1), ship.y + 1, -ship.speed)
    } else if (proposedX < 0) {
      SState(-proposedX-1, ship.y + 1, -ship.speed)
    } else {
      SState(proposedX, ship.y, ship.speed)
    }
  }

  def update(l: Set[SState]): Set[SState] = {
    val newl = l map updateShip
    val targets = newl filter (sh => sh.x == player.x)
    if (targets.size == 0)
      newl
    else {
      val killed = targets.toVector.sortBy(sh => sh.y).reverse.head
      //println("Killed "+killed)
      newl.-(killed)
    }
  }

  def write(l: Set[SState]): Unit = {
    (0 until width).foreach(x => print("="))
    println("")
    (0 until height).foreach { y =>
      (0 until width).foreach { x => {
        if (l.filter(s => (s.x == x)&&(s.y == y)).size > 0)
          print("W")
        else if ((x == player.x)&&(y == player.y))
          print("^")
        else
          print(" ")
      }
      }
        println("")
    }
    (0 until width).foreach(x => print("-"))
    println("")
    println("")
    Thread.sleep(1000)
    }

  def main(args: Array[String]): Unit = {

    val ships = (0 until width).toSet.map((x: Int) => SState(x,0,1)) ++
      (0 until width).toSet.map((x: Int) => SState(x,1,1)) ++
      (0 until width).toSet.map((x: Int) => SState(x,2,-1)) ++
      (0 until width).toSet.map((x: Int) => SState(x,3,-2)) ++
      (0 until width).toSet.map((x: Int) => SState(x,4, 3))

    val shipStream = Stream.iterate(ships)(s => update(s))

    shipStream.foreach(write(_)) 

  }

}
