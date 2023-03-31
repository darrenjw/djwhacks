/*
nspl-test.scala
Stub for Scala Cats code
*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import org.nspl.*
import org.nspl.awtrenderer.*

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

  val df = csv.CsvParser.parseFileWithHeader[Double](java.io.File("mcmc.csv"))

  }

