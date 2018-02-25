package my.scalafx

import scala.collection.JavaConverters._
import scalafx.animation.AnimationTimer
import scalafx.application.JFXApp
import scalafx.application.JFXApp.PrimaryStage
import scalafx.scene.Scene
import scalafx.scene.image.{ImageView, WritableImage}
import scalafx.scene.paint.Color._
import scalafx.scene.paint._
import scalafx.scene.text.Text

import cats._
import cats.implicits._

import scala.collection.parallel.immutable.ParVector
case class Image[T](w: Int, h: Int, data: ParVector[T]) {
  def apply(x: Int, y: Int): T = data(x*h+y)
  def map[S](f: T => S): Image[S] = Image(w, h, data map f)
  def updated(x: Int, y: Int, value: T): Image[T] =
    Image(w,h,data.updated(x*h+y,value))
}

case class PImage[T](x: Int, y: Int, image: Image[T]) {
  def extract: T = image(x, y)
  def map[S](f: T => S): PImage[S] = PImage(x, y, image map f)
  def coflatMap[S](f: PImage[T] => S): PImage[S] = PImage(
    x, y, Image(image.w, image.h,
    (0 until (image.w * image.h)).toVector.par.map(i => {
      val xx = i / image.h
      val yy = i % image.h
      f(PImage(xx, yy, image))
    })))
 def up: PImage[T] = {
    val py = y-1
    val ny = if (py >= 0) py else (py + image.h)
    PImage(x,ny,image)
  }
  def down: PImage[T] = {
    val py = y+1
    val ny = if (py < image.h) py else (py - image.h)
    PImage(x,ny,image)
  }
  def left: PImage[T] = {
    val px = x-1
    val nx = if (px >= 0) px else (px + image.w)
    PImage(nx,y,image)
  }
  def right: PImage[T] = {
    val px = x+1
    val nx = if (px < image.w) px else (px - image.w)
    PImage(nx,y,image)
  }
}

object ScalaFXHelloWorld extends JFXApp {

  val w = 600
  val h = 500
  val beta = 0.45

  implicit val pimageComonad = new Comonad[PImage] {
    def extract[A](wa: PImage[A]) = wa.extract
    def coflatMap[A,B](wa: PImage[A])(f: PImage[A] => B): PImage[B] =
      wa.coflatMap(f)
    def map[A,B](wa: PImage[A])(f: A => B): PImage[B] = wa.map(f)
  }

  val pim0 = PImage(0,0,Image(w,h,Vector.fill(w*h)(if (math.random > 0.5) 1 else -1).par))

  def gibbsKernel(pi: PImage[Int]): Int = {
    val sum = pi.up.extract+pi.down.extract+pi.left.extract+pi.right.extract
    val p1 = math.exp(beta*sum)
    val p2 = math.exp(-beta*sum)
    val probplus = p1/(p1+p2)
    if (math.random < probplus) 1 else -1
  }

  def oddKernel(pi: PImage[Int]): Int =
    if ((pi.x+pi.y) % 2 != 0) pi.extract else gibbsKernel(pi)
  def evenKernel(pi: PImage[Int]): Int =
    if ((pi.x+pi.y) % 2 == 0) pi.extract else gibbsKernel(pi)

  def pims = Stream.iterate(pim0)(_.coflatMap(oddKernel).
    coflatMap(evenKernel))

  def sfxis = pims map (pi => I2SFXWI(pi.image))

  def I2SFXWI(im: Image[Int]): WritableImage = {
    val wi = new WritableImage(im.w,im.h)
    val pw = wi.pixelWriter
    (0 until im.w) foreach (i =>
      (0 until im.h) foreach (j =>
        pw.setColor(i, j, Color.gray(if (im(i,j)==1) 1 else 0))
        )
    )
    wi
  }

  stage = new PrimaryStage {
    title = "My ScalaFX test"
    scene = new Scene {
      fill = Color.rgb(0,0,0)
      var is = sfxis
      val iv = new ImageView(is.head)
      content = iv
      val delay = 1 // nano seconds, so 1e9 is 1 second 
      var lastUpdate=0L
      val timer = AnimationTimer(t => {
        if (t-lastUpdate > delay) {
          iv.image = is.head
          is = is.tail
          lastUpdate = t
        }
      })
      timer.start
    }

  }
}
