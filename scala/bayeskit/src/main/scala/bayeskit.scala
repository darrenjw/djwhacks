package bayeskit

object bayeskit {

  def main(args: Array[String]): Unit = {
    println("Starting...")
    val its=if (args.length==0) 10 else args(0).toInt
    println("Running for "+its+" iters:")
    lvsim.runModel(its)
    println("Done.")
  }

}