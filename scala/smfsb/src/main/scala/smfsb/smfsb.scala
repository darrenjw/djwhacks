/*
smfsb.scala

Replicate functionality from my "smfsb" R package in Scala

 */

package smfsb

object SmfsbApp {

  def main(args: Array[String]): Unit = {
    println("hello")
    if (args.length == 1) {
      val its = args(0).toInt
      import LvPmmh._
      runModel(its)
    } else {
      println("sbt \"run <its>\"")
    }
    println("goodbye")
  }

}

/* eof */

