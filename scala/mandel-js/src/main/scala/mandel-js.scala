/*
scalajs-example.scala

https://www.scala-js.org/
https://www.scala-js.org/doc/


 */

import scala.scalajs.js.annotation.{JSExport,JSExportTopLevel}
import org.scalajs.dom
import org.scalajs.dom.html

import spire.*
import spire.math.*
import spire.implicits.*

@JSExportTopLevel("JsApp")
object ScalaJsTest:

  val cSize = 768
  val mSize = 3.0
  val z0 = Complex(-2.0, -1.5)
  val maxIt = 150

  def mandel(c: Complex[Double]): Int =
    val z0 = Complex(0.0,0.0)
    @annotation.tailrec
    def go(z: Complex[Double], n: Int): Int =
      if (n >= maxIt)
        -1
      else
        val newz = z*z + c
        if (newz.abs > 2)
          n
        else
          go(newz, n+1)
    go(z0, 0)

  def render(ctx: dom.CanvasRenderingContext2D): Unit =
    (0 until cSize).foreach(x =>
      (0 until cSize).foreach(y =>
        val c = z0 + Complex(mSize*x/cSize, mSize*y/cSize)
        val its = mandel(c)
        val shade = math.round(255*math.log(its+1)/math.log(maxIt+1)).toInt
        val fs = s"rgb($shade, $shade, $shade)"
        ctx.fillStyle = if (its == -1) "rgb(30, 30, 30)" else fs
        ctx.fillRect(x, y, 1, 1)
      )
    )

  @JSExport
  def main(canvas: html.Canvas): Unit =
    val ctx = canvas.getContext("2d")
                    .asInstanceOf[dom.CanvasRenderingContext2D]
    def clear() =
      ctx.fillStyle = "lightgrey"
      ctx.fillRect(0, 0, cSize, cSize)
    def run =
      clear()
      render(ctx)
    run



// eof

