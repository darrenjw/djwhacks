/*
monte-carlo6.scala

scalac monte-carlo6.scala
time scala MonteCarlo

*/

import java.util.concurrent.ThreadLocalRandom
import scala.math.exp

object MonteCarlo {

  def main(args: Array[String]) = {
    println("Hello")
    val iters=1000000
    val sums=(1 to iters).toList.par map {x => ThreadLocalRandom.current().nextDouble()} map {x => exp(-x*x)}
    val result=sums.reduce(_+_)
    println(result/iters)
    println("Goodbye")
  }

}
