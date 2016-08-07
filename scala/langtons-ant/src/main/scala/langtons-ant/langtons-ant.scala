/*
langtons-ant.scala

https://en.wikipedia.org/wiki/Langton%27s_ant

 */

import java.awt.image.BufferedImage

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

  def stateStreamManual(s: State): Stream[State] = s #:: stateStream(nextState(s))

  def stateStream(s: State): Stream[State] = Stream.iterate(s)(nextState(_))

  def thinStream[T](s: Stream[T],th: Int): Stream[T] = {
    val ss = s.drop(th)
    ss.head #:: thinStream(ss,th)
    }

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

  def biResize(img: BufferedImage,newW: Int,newH: Int): BufferedImage = {
    val tmp = img.getScaledInstance(newW, newH, java.awt.Image.SCALE_REPLICATE);
    val dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);
    val g2d = dimg.createGraphics();
    g2d.drawImage(tmp, 0, 0, null);
    g2d.dispose();
    dimg
    }

  def main(args: Array[String]): Unit = {
    println("starting")
    val ss = stateStream(State(300))
    val is = ss.map(s => s.img)
    val bis = is.map(img2Image(_))
    val fin = bis.drop(20000).head
    javax.imageio.ImageIO.write(fin, "png", new java.io.File("img.png"))
    println("done")
  }

}

/* eof */

