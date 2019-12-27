/*
evilplot-examples.scala

EvilPlot examples

*/

object EvilPlotExamples {

  import scala.util.Random
  import com.cibo.evilplot._
  import com.cibo.evilplot.plot._
  import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
  import com.cibo.evilplot.numeric._
  import com.cibo.evilplot.plot.renderers.PointRenderer

  def scatterExample() = {
    val points = Seq.fill(150) {
      Point(Random.nextDouble(), Random.nextDouble())
    } :+ Point(0.0, 0.0)
    val years = Seq.fill(150)(Random.nextDouble()) :+ 1.0
    ScatterPlot(
      points,
      //pointRenderer = Some(PointRenderer.depthColor((i: Int) => years(i), 0.0, 1.0, None, None))
)
      .standard()
      .xLabel("x")
      .yLabel("y")
      .trend(1, 0)
      .rightLegend()
      .render()
  }


  def main(args: Array[String]): Unit = {

    displayPlot(scatterExample())



  }


}

// eof
