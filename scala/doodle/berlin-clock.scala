/*

Draw a berlin clock using "doodle"

*/

import doodle.core._
import doodle.syntax._
import doodle.jvm._
import doodle.examples._
import doodle.jvm.Java2DCanvas._
import doodle.backend.StandardInterpreter._

import java.util.Calendar


object BerlinClockApp
{

  case class Clock(evenSec: Boolean,fiveHr: Int, oneHr: Int, fiveMin: Int, oneMin: Int) {
    override def toString = {
      val r1 = if (evenSec) "*" else "O"
      val r2 = "*"*fiveHr+"-"*(4-fiveHr)
      val r3 = "*"*oneHr+"-"*(4-oneHr)
      val r4 = "*"*fiveMin+"-"*(11-fiveMin)
      val r5 = "*"*oneMin+"-"*(4-oneMin)
      r1+"\n"+r2+"\n"+r3+"\n"+r4+"\n"+r5
    }
  }

  object Clock {
    def apply(hr: Int, min: Int, sec: Int): Clock = 
     Clock(sec%2 == 0, hr/5, hr%5, min/5, min%5)
    def apply(dt: Calendar): Clock =
      Clock(dt.get(Calendar.HOUR_OF_DAY),
            dt.get(Calendar.MINUTE),
            dt.get(Calendar.SECOND))
    def apply(): Clock = Clock(Calendar.getInstance())
  }

  def clockImage(cl: Clock): Image = {
    println(cl)
    val drk=0.4
    val sc=3
    val row1 = (Circle(sc*30) 
                fillColor Color.red.darken((if (cl.evenSec) 0.0 else drk).normalized) 
                lineWidth 3)
    val row2 = allBeside((0 to 3) map {x => (Rectangle(sc*30,sc*20) 
                fillColor Color.red.darken((if (x>=cl.fiveHr) drk else 0.0).normalized) 
                lineWidth 3)})
    val row3 = allBeside((0 to 3) map {x => (Rectangle(sc*30,sc*20) 
                fillColor (Color.red.darken((if (x>=cl.oneHr) drk else 0.0).normalized))
                lineWidth 3)})
    val row4 = allBeside((0 to 10) map {x => (Rectangle(sc*11,sc*20) 
                fillColor (if (x%3 == 2) Color.red else 
                Color.yellow).darken((if (x>=cl.fiveMin) drk else 0.0).normalized) 
                lineWidth 3)})
    val row5 = allBeside((0 to 3) map {x => (Rectangle(sc*30,sc*20) 
                fillColor Color.yellow.darken((if (x>=cl.oneMin) drk else 0.0).normalized) 
                lineWidth 3)})
    row1 above row2 above row3 above row4 above row5
  }


  def main(args: Array[String]): Unit = {
    (1 to 1000) foreach { x => Thread.sleep(1000); clockImage(Clock()).draw }
  }

}

// eof


