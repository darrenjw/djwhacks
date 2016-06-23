/*
langtons-ant.scala

https://en.wikipedia.org/wiki/Langton%27s_ant

 */

case class Img(size: Int, data: Vector[Boolean]) {

  def apply(x: Int, y: Int): Boolean = data(y * size + x)

  def updated(x: Int, y: Int, p: Boolean): Img = Img(size, data.updated(y * size + x, p))

  override def toString: String = {
    val s = data map { p => if (p) "t" else "f" }
    s.mkString(" ")
  }

}

object Img {

  def apply(size: Int) = new Img(size, Vector.fill(size * size)(false))

}

case class Ant(x: Int, y: Int, d: Int) {

  def move: Ant = d match {
    case 0 => Ant(x + 1, y, d)
    case 1 => Ant(x, y + 1, d)
    case 2 => Ant(x - 1, y, d)
    case _ => Ant(x, y - 1, d)
  }

}

case class State(ant: Ant, img: Img)

object State {

  def apply(size: Int): State = State(Ant(size / 2, size / 2, 0), Img(size))

}

object LangtonsAnt {

  def nextState(s: State): State = {
    val size = s.img.size
    val ant = s.ant.move
    val wAnt = Ant(ant.x % size, ant.y % size, ant.d)

    s
  }

  def main(args: Array[String]): Unit = {
    println("hi")
    val im = Img(5).updated(2, 2, true)
    println(im)
    println("bye")
  }

}

/* eof */

