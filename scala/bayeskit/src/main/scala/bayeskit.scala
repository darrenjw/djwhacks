package bayeskit

object bayeskit {

  import sim._
  import lvsim.stepLV
  import breeze.stats.distributions._
  import pmmh._
  import pfilter._
  import java.io.{File,PrintWriter,OutputStreamWriter}
  
  
  def main(args: Array[String]): Unit = {
    println("hello")
    val state = stepLV(Vector(100, 50), 0, 10, Vector(1.0, 0.005, 0.6))
    println(state.toString)
    val ts = simTs(Vector(100, 50), 0, 100, 0.1, stepLV, Vector(1.0, 0.005, 0.6))
    println(ts)
    def simPrior(n: Int, t: Time, th: Parameter): Vector[State] = {
      val prey = new Poisson(100.0).sample(n).toVector
      val predator = new Poisson(50.0).sample(n).toVector
      prey.zip(predator) map { x => Vector(x._1, x._2) }
    }
    def obsLik(s: State, o: Observation, th: Parameter): Double = {
      new Gaussian(s(0), 10.0).pdf(o(0))
    }
    val truth = simTs(Vector(100, 50), 0, 30, 2.0, stepLV, Vector(1.0, 0.005, 0.6))
    val data = truth map { x => (x._1, Vector(x._2(0).toDouble)) }
    //val mll = pfMLLik(100, simPrior, 0.0, stepLV, obsLik, data)
    val mll = pfMLLikPar(100, simPrior, 0.0, stepLV, obsLik, data)
    val s=new PrintWriter(new File("mcmc-out.csv" ))
    //val s=new OutputStreamWriter(System.out)
    s.write("th1,th2,th3\n")
    val pmmhOutput=runPmmh(s,1000,Vector(1.0, 0.005, 0.6),mll)
    s.close
    println("goodbye")
  }

}