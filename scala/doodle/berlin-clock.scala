/*

Draw a berlin clock using "doodle"

*/

import doodle.core._
import doodle.syntax._
import doodle.jvm._
import doodle.examples._
import doodle.jvm.Java2DCanvas._
import doodle.backend.StandardInterpreter._

object BerlinClockApp
{

  def main(args: Array[String]): Unit = {
    val row1 = (Circle(30) 
                fillColor Color.red lineWidth 3)
    val row2 = allBeside((0 to 3) map {x => (Rectangle(30,20) 
                fillColor Color.red lineWidth 3)})
    val row3 = allBeside((0 to 3) map {x => (Rectangle(30,20) 
                fillColor (Color.red.darken(0.3.normalized)) lineWidth 3)})
    val row4 = allBeside((0 to 10) map {x => (Rectangle(11,20) 
                fillColor (if (x%3 == 2) Color.red else Color.yellow) lineWidth 3)})
    val row5 = allBeside((0 to 3) map {x => (Rectangle(30,20) 
                fillColor Color.yellow lineWidth 3)})
    val im = row1 above row2 above row3 above row4 above row5
    im.draw
  }

}

// eof


