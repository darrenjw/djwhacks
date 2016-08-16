/*
langtons-ant.scala

https://en.wikipedia.org/wiki/Langton%27s_ant

 */

import java.awt.image.BufferedImage
import scalaview.SwingImageViewer
import scalaview.Utils._

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
    val wa = Ant((ant.x + size) % size, (ant.y + size) % size, ant.d)
    if (s.img(wa.x, wa.y)) State(Ant(wa.x, wa.y, (wa.d + 1) % 4), s.img.updated(wa.x, wa.y, false))
    else State(Ant(wa.x, wa.y, (wa.d + 3) % 4), s.img.updated(wa.x, wa.y, true))
  }

  def stateStream(s: State): Stream[State] = Stream.iterate(s)(nextState(_))

  def img2Image(img: Img): BufferedImage = {
    val canvas = new BufferedImage(img.size, img.size, BufferedImage.TYPE_BYTE_BINARY)
    val wr = canvas.getRaster
    for (x <- 0 until img.size) {
      for (y <- 0 until img.size) {
        wr.setSample(x, y, 0, if (img(x, y)) 0 else 1)
      }
    }
    canvas
  }

  def main(args: Array[String]): Unit = {
    val ssize = 250
    val stretch = 5
    val is = thinStream(stateStream(State(ssize)), 100).
      map(s => img2Image(s.img)).
      map(biResize(_, ssize * stretch, ssize * stretch))
    SwingImageViewer(is)
  }

}

/* eof */

