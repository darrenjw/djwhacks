import scala.scalajs.js.JSApp
import scala.scalajs.js.annotation.{JSExport,JSExportTopLevel}
import org.scalajs.dom
import org.scalajs.dom.html
import org.scalajs.dom.html.Canvas
import dom.{ document, window }

case class P(x: Int, y: Int)

@JSExportTopLevel("JsApp")
object MainApp extends JSApp {

  val blockSize = 20
  val width = 10
  val height = 8
  val fullWidth = width*blockSize
  val fullHeight = width*blockSize

  val path = List(P(1,1),P(1,2),P(1,3),P(1,4),P(2,2))

  def main(): Unit = {

    val canvas = document.createElement("canvas").asInstanceOf[Canvas]
    canvas.width = fullWidth
    canvas.height = fullHeight

    val ctx = canvas.getContext("2d")
                    .asInstanceOf[dom.CanvasRenderingContext2D]

    val p = document.createElement("p")
    val text = document.createTextNode("Hello!")
    p.appendChild(text)
    document.body.appendChild(p)

    def clear() = {
      ctx.fillStyle = "lightgrey"
      ctx.fillRect(0, 0, fullWidth, fullHeight)
    }

    def run = {
      clear
      ctx.fillStyle = "black"
      ctx.fillRect(20,20,40,40)
    }

    dom.window.setInterval(() => run, 10)


  }

}
