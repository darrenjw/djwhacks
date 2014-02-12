package bayeskit

object bayeskit {

  import sim._
  import lvsim.stepLV

  def main(args: Array[String]): Unit = {
    println("hello")
    val state = stepLV(Vector(100, 50), 0, 10, Vector(1.0, 0.005, 0.6))
    println(state.toString)
    val ts = simTs(Vector(100, 50), 0, 100, 0.1, stepLV, Vector(1.0, 0.005, 0.6))
    println(ts)
    println("goodbye")
  }

}