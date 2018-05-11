/*
evilplot.scala

Test of the EvilPlot library and using it in conjunction with scala-view

 */

object EvilPlotTest {

  import com.cibo.evilplot.plot._
  import com.cibo.evilplot.colors._
  import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
  import com.cibo.evilplot.numeric.Point
  import java.awt.Image.SCALE_SMOOTH
  import scalaview.Utils._

  def simpleScatter: Unit = {
    val data = Seq.tabulate(100) { i =>
      Point(i.toDouble, scala.util.Random.nextDouble())
    }
    val plot = ScatterPlot(data)
      .xAxis()
      .yAxis()
      .frame()
      .xLabel("x")
      .yLabel("y")
      .render()
    val plotImage = plot.asBufferedImage
    //scalaview.SwingImageViewer(biResize(plotImage,1000,800,SCALE_SMOOTH))
    scalaview.SfxImageViewer(biResize(plotImage,1000,800,SCALE_SMOOTH))
  }

  def simpleLineGraph: Unit = {
    val data = Seq.tabulate(100) { i =>
      Point(i.toDouble, scala.util.Random.nextDouble())
    }
    val plot = LinePlot.series(data, "Line graph", HSL(210, 100, 56))
      .xAxis()
      .yAxis()
      .frame()
      .xLabel("x")
      .yLabel("y")
      .render()
    scalaview.SfxImageViewer(biResize(plot.asBufferedImage,1000,800,SCALE_SMOOTH))
  }

  def simplePlotStream: Unit = {
    val data = Seq.tabulate(100) { i =>
      Point(i.toDouble, scala.util.Random.nextDouble())
    }
    val dataStream = data.toStream
    val cumulStream = dataStream.scanLeft(Nil: List[Point])((l,p) => p :: l).drop(1)
    def dataToImage(data: List[Point]) = {
      val plot = LinePlot.series(data, "Line graph", HSL(210, 100, 56))
        .xAxis()
        .yAxis()
        .frame()
        .xLabel("x")
        .yLabel("y")
        .render()
      plot.asBufferedImage
    }
    val plotStream = cumulStream map (d => biResize(dataToImage(d),1000,800,SCALE_SMOOTH))
    scalaview.SfxImageViewer.bi(plotStream, 100000, autoStart=true)
  }

  def main(args: Array[String]): Unit = {
    println("Hi")
    //simpleScatter
    simpleLineGraph
    simplePlotStream
    println("Bye")
  }

}

// eof

