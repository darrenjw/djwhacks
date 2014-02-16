/*
monte-carlo.scala

scalac monte-carlo.scala
time scala MonteCarlo

*/

import scala.util.Random.nextDouble
import scala.math.exp
import scala.annotation.tailrec

object MonteCarlo {

  @tailrec
  def sum(its: Long,acc: Double): Double = {
    if (its==0) 
      (acc)
    else {
      val u=nextDouble
      sum(its-1,acc+exp(-u*u))
    }
  }

  def main(args: Array[String]) = {
    println("Hello")
    val iters=100000000
    val result=sum(iters,0.0)
    println(result/iters)
    println("Goodbye")
  }

}
