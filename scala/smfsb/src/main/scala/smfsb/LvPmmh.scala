/*
LvPmmh.scala

Actual LV PMMH example from my book (prey only, known noise)
Mainly to check everything is working correctly...

 */

package smfsb

object LvPmmh {

  import breeze.stats.distributions._
  import breeze.linalg.DenseVector
  import scala.annotation.tailrec
  import scala.collection.immutable.{Vector => IVec}
  import scala.io.Source
  import java.io.{File, PrintWriter, OutputStreamWriter}
  import Types._
  import SpnExamples._
  import Mll.pfMll
  import Pmmh.runPmmh

  def simPrior(th: LvParameter)(n: Int, t: Time): IVec[IntState] = {
    val prey = new Poisson(50.0).sample(n).toVector
    val predator = new Poisson(100.0).sample(n).toVector
    prey.zip(predator) map { x => DenseVector(x._1, x._2) }
  }

  def obsLik(th: LvParameter)(s: IntState, o: DoubleState): LogLik = {
    //Gaussian(s(0).toDouble, 10.0).logPdf(o(0)) // doesn't work...
    val sigma = 10.0
    val mu: Double = s.copy.apply(0).toDouble // problem line! Seem to have to copy?!
    val x: Double = o(0)
    val q = (x - mu) / sigma
    val ll: Double = -math.log(sigma) - 0.5 * q * q
    ll
  }

  def runModel(its: Int): Unit = {
    val rawData = Source.fromFile("LVpreyNoise10.txt").getLines
    val data = ((0 to 30 by 2).toList zip rawData.toList) map { x => (x._1.toDouble, DenseVector(x._2.toDouble)) }
    val mll = pfMll(150, simPrior, 0.0, stepLv, obsLik, data)
    val s = new PrintWriter(new File("LVPN10-Pmmh50k.csv"))
    // val s=new OutputStreamWriter(System.out)
    s.write("th0,th1,th2\n")
    //s.write(((0 to 30 by 2) map { n => "x" + n + ",y" + n }).mkString(",") + "\n")
    val pmmhOutput = runPmmh(s, its, lvparam, mll)
    s.close
  }

  def main(args: Array[String]): Unit = {
    runModel(50000)
  }

}

/* eof */
