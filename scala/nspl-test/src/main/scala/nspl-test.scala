/*
nspl-test.scala

Test of nspl (and resurrected saddle)

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import org.nspl.*
import org.nspl.awtrenderer.*
import org.nspl.data.HistogramData
import org.nspl.saddle.*
import org.saddle.*

import scala.util.Random.nextDouble
import java.io.File

object NsplApp extends IOApp.Simple:

  val l = List(1,2) |+| List(3,4)

  def run = IO{

  val someData = 
    0 until 100 map (_ => nextDouble() -> nextDouble())

  val plot = xyplot(someData)(
              par.withMain("Main label")
              .withXLab("x axis label")
              .withYLab("y axis label")
            )

  //renderToByteArray(plot.build, width=2000)
  //show(plot.build)
  pngToFile(File("test.png"), plot.build, width=1500)

  // Try plotting some MCMC output
  val dfe = csv.CsvParser.parseFileWithHeader[Double](File("mcmc.csv"),
    recordSeparator="\n") // Unix line endings...
  println(dfe)
  val df = dfe.toOption.get // Yikes!

  val p1 = boxplot(df.colSlice(0,10), boxColor=Color.red)(par)
  pngToFile(File("bp.png"), p1.build, width=2000)

  val p2 = xyplot(HistogramData(df.colAt(0).toVec.toSeq, 50) -> bar(
    strokeColor=Color.red, fill=Color.grey2))(par)
  pngToFile(File("hist.png"), p2.build, width=1000)

  val p3 = xyplot((df.colAt(0).toVec, line(color=Color.red)))(par)
  pngToFile(File("trace.png"), p3.build, width=1500)


  }

