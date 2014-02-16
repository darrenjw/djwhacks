/*
monte-carlo4.scala

scalac monte-carlo4.scala
time scala MonteCarlo

*/

import java.util.concurrent.ThreadLocalRandom
import scala.math.exp
import scala.annotation.tailrec

object MonteCarlo {

  @tailrec
  def sum(its: Long,acc: Double): Double = {
    if (its==0) 
      (acc)
    else {
      val u=ThreadLocalRandom.current().nextDouble()
      sum(its-1,acc+exp(-u*u))
    }
  }

  def main(args: Array[String]) = {
    println("Hello")
    val N=args(0).toInt
    val iters=100000000
    val its=iters/N
    val sums=(1 to N).toList.par map {x => sum(its,0.0)}
    val result=sums.reduce(_+_)
    println(result/iters)
    println("Goodbye")
  }

}
