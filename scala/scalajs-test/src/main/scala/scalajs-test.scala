import scala.scalajs.js.annotation.{JSExport,JSExportTopLevel}
import org.scalajs.dom
import org.scalajs.dom.html

@JSExportTopLevel("JsApp")
object ScalaJsTest {

  val cSize = 512
  val size = 20

  case class Square(x: Int, y: Int, vx: Int, vy: Int, col: String) {

    def render(ctx: dom.CanvasRenderingContext2D): Unit = {
      ctx.fillStyle = col
      ctx.fillRect(x, y, size, size)
    }

  }

  val squares = List(
    Square(0,0,4,3,"blue"),
    Square(200,0,2,-1,"red"),
    Square(100,50,1,2,"green")
  )

  def nextSquare(sq: Square): Square = {
    val can = sq.copy(x=sq.x+sq.vx,y=sq.y+sq.vy)
    if (can.x + size >= cSize) sq.copy(vx = -sq.vx)
    else if (can.y + size >= cSize) sq.copy(vy = -sq.vy)
    else if (can.x <= 0) sq.copy(vx = -sq.vx)
    else if (can.y <= 0) sq.copy(vy = -sq.vy)
    else can
  }

  def squaresS = Stream.iterate(squares)(_ map nextSquare)

  @JSExport
  def main(canvas: html.Canvas): Unit = {

    val ctx = canvas.getContext("2d")
                    .asInstanceOf[dom.CanvasRenderingContext2D]

    def clear() = {
      ctx.fillStyle = "lightgrey"
      ctx.fillRect(0, 0, cSize, cSize)
    }

    var sqrs = squaresS

    def run = {
      val sq = sqrs.head
      sqrs = sqrs.tail
      clear
      sq map (_.render(ctx))
    }

    dom.window.setInterval(() => run, 10)

  }

}

