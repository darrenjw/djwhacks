/*
assert-test.scala

Test assertions and exclusion at compile time
*/

object AssertTest {

  def main(args: Array[String]): Unit = {
    assert(args.length == 0)
    println("hello")
    time {
      (1 to 1000000000).foreach(i => assert(args.length != i))
    }
    println("bye")
  }

  // function for timing
  def time[A](f: => A) = {
    val s = System.nanoTime
    val ret = f
    println("time: "+(System.nanoTime-s)/1e6+"ms")
    ret 
  }


}

/* eof */
