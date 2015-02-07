/*
wisp.scala

Test of "wisp"

*/

import com.quantifind.charts.Highcharts._

object WispTest {

  def main(args: Array[String]) = {
    println("Hello")
    areaspline(List(1, 2, 3, 4, 5), List(4, 1, 3, 2, 6))
    areaspline(List(1, 2, 3, 4, 5), List(4, 1, 3, 2, 6))
    areaspline(List(1, 2, 3, 4, 5), List(4, 1, 3, 2, 6))
    delete
    stopServer
    println("Done")
  }

}



