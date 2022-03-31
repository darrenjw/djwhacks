/*
scalajs-example.scala

https://www.scala-js.org/
https://www.scala-js.org/doc/


 */

import scala.scalajs.js.annotation.{JSExport,JSExportTopLevel}
import org.scalajs.dom

@JSExportTopLevel("Plotl")
object PlotlyStuff:

  import org.openmole.plotlyjs.*
  import org.openmole.plotlyjs.all.*
  import org.openmole.plotlyjs.PlotlyImplicits.{given,*}
  import org.openmole.plotlyjs.plotlyConts.*
  import com.raquo.laminar.api.L.{given,*}

  import scala.scalajs.js.JSConverters.*

  val plotDiv = div()

  val layout = Layout
    .title("My line plot")
    .showlegend(true)
    .xaxis(axis.title("Time"))
    .yaxis(axis.title("Production"))

  val data = linechart.lines

  val ref = (1 to 15).toJSArray

  val dataRef = data
    .x((0 to 14).toJSArray)
    .y(ref)
    .marker(marker.symbol(square).color(all.color.rgb(180, 0, 0)).size(12.0))
    .name("Reds")


  val dataN = (for (i <- 1 to 6) yield {
    data
      .x((0 to 14).toJSArray)
      .y(ref.map { x => x + 0.3 * 2 - 1 }.toJSArray)
      .marker(marker.size(10.0).color(all.color.rgb(200, 136, 170)))
      ._result
  }).toJSArray


  val config = Config.displayModeBar(false)

  @JSExport
  def plot(hDiv: dom.html.Div): Unit =
    Plotly.newPlot(plotDiv.ref,
      dataN :+ dataRef._result,
      layout,
      config)
    render(hDiv, plotDiv)


@JSExportTopLevel("JsApp")
object ScalaJsTest:

  val cSize = 512
  val size = 20

  case class Square(x: Int, y: Int, vx: Int, vy: Int, col: String):
    def render(ctx: dom.CanvasRenderingContext2D): Unit =
      ctx.fillStyle = col
      ctx.fillRect(x, y, size, size)

  val squares = List(
    Square(0,0,4,3,"blue"),
    Square(200,0,2,-1,"red"),
    Square(100,50,1,2,"green")
  )

  def nextSquare(sq: Square): Square =
    val can = sq.copy(x=sq.x+sq.vx,y=sq.y+sq.vy)
    if (can.x + size >= cSize) sq.copy(vx = -sq.vx)
    else if (can.y + size >= cSize) sq.copy(vy = -sq.vy)
    else if (can.x <= 0) sq.copy(vx = -sq.vx)
    else if (can.y <= 0) sq.copy(vy = -sq.vy)
    else can

  def squaresS = LazyList.iterate(squares)(_ map nextSquare)

  @JSExport
  def main(canvas: dom.html.Canvas): Unit =
    val ctx = canvas.getContext("2d")
                    .asInstanceOf[dom.CanvasRenderingContext2D]
    def clear() =
      ctx.fillStyle = "lightgrey"
      ctx.fillRect(0, 0, cSize, cSize)
    var sqrs = squaresS
    def run =
      val sq = sqrs.head
      sqrs = sqrs.tail
      clear()
      sq map (_.render(ctx))
    dom.window.setInterval(() => run, 10)

import org.scalajs.dom.raw.HTMLInputElement

@JSExportTopLevel("JsFun")
object JsFunction:

  @JSExport
  def factor(form: dom.html.Form): Boolean =
    val facfield = form("factorfield").asInstanceOf[HTMLInputElement]
    val numS = facfield.value
    val num = numS.toInt
    val fac = factor(num)
    if (num == fac)
      dom.window.alert(s"$numS is prime!")
    else
      val facS = fac.toString
      val rem = num / fac
      facfield.value = rem.toString
      dom.window.alert(s"Smallest factor of $numS is: $facS")
    false

  @annotation.tailrec
  def factor(n: Int, f: Int = 2): Int =
    if (n % f == 0) f
    else if (f >= n) n
    else factor(n,f+1)



// eof

