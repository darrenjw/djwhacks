/*
futures.scala
pure scala - no build file needed - just run with:

sbt run

*/

object Main {

 import scala.concurrent.{Future,ExecutionContext}
 import ExecutionContext.Implicits.global

 def main(args: Array[String]): Unit = {

  val fut1=Future{
    Thread.sleep(10000)
    1
  }

  val fut2=Future{
    Thread.sleep(5000)
    2
  }

  val futSum=for {
    v1 <- fut1
    v2 <- fut2
  } yield (v1+v2)

  futSum.map{s => println("futSum completes to: "+s)}

  println("Sleeping for 20 seconds...")
  Thread.sleep(20000)
  println("Finished sleeping.")

 }

}


// eof


