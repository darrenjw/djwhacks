package bayeskit

object bayeskit {

  import sim._
  import lvsim._
  import breeze.stats.distributions._
  import pmmh._
  import pfilter._
  import scala.io.Source
  import java.io.{ File, PrintWriter, OutputStreamWriter }

  def main(args: Array[String]): Unit = {
    println("Starting...")
    val its=if (args.length==0) 10 else args(0).toInt
    println("Running for "+its+" iters:")
    def simPrior(n: Int, t: Time, th: Parameter): Vector[LvState] = {
      val prey = new Poisson(50.0).sample(n).toVector
      val predator = new Poisson(100.0).sample(n).toVector
      prey.zip(predator) map { x => LvState(x._1, x._2) }
    }
    def obsLik(s: LvState, o: Observation, th: Parameter): Double = {
      new Gaussian(s.prey, 10.0).pdf(o(0))
    }
    val rawData = Source.fromFile("LVpreyNoise10.txt").getLines
    val data = ((0 to 30 by 2).toList zip rawData.toList) map { x => (x._1.toDouble, Vector(x._2.toDouble)) }
    val mll = pfPropPar(100, simPrior, 0.0, stepLV, obsLik, data)
    val s = new PrintWriter(new File("mcmc-out.csv"))
    // val s=new OutputStreamWriter(System.out)
    s.write("th1,th2,th3,")
    s.write(((0 to 30 by 2) map { n => "x" + n +",y"+ n }).mkString(",") + "\n")
    val pmmhOutput = runPmmhPath(s, its, LvParameter(1.0, 0.005, 0.6), mll,peturb)
    s.close
    println("Done.")
  }

}