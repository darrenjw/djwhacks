import scala.scalajs.js.annotation.{JSExport,JSExportTopLevel}
import org.scalajs.dom
import org.scalajs.dom.html
import scala.util.Random

@JSExportTopLevel("JsApp")
object ScalaJsTest {

  @JSExport
  def main(canvas: html.Canvas): Unit = {
    val ctx = canvas.getContext("2d")
                    .asInstanceOf[dom.CanvasRenderingContext2D]

    val cSize = 512

    def clear() = {
      ctx.fillStyle = "lightgrey"
      ctx.fillRect(0, 0, cSize, cSize)
    }

    var x = 0
    var y = 0
    val size = 20
    var vx = 4
    var vy = 3

    def run = {
      x=x+vx
      y=y+vy
      if (x+size >= cSize) vx = -vx
      if (y+size >= cSize) vy = -vy
      if (x <= 0) vx = -vx
      if (y <= 0) vy = -vy
      clear
      ctx.fillStyle = "black"
      ctx.fillRect(x, y, size, size)
    }

    dom.window.setInterval(() => run, 50)
  }
}
