/*
wisp.scala

Test of "wisp"

*/

import com.quantifind.charts.Highcharts._

object WispTest {

  def main(args: Array[String]) = {
    println("Hello")
    help
    areaspline(List(1, 2, 3, 4, 5), List(4, 1, 3, 2, 6))
    areaspline(List(1, 2, 3, 4, 5), List(4, 1, 3, 2, 6))
    areaspline(List(1, 2, 3, 4, 5), List(4, 1, 3, 2, 6))
    delete
    deleteAll
    line((0 until 5).map(x => x -> x*x))
    stopServer
    println("Done")
  }

}



