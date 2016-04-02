/*
ArPmmh.scala

PMMH for the AutoReg model

Also a function for simulating a sample dataset from the AR model

 */

package smfsb

object ArPmmh {

  import breeze.stats.distributions._
  import breeze.linalg.DenseVector
  import scala.annotation.tailrec
  import scala.collection.immutable.{Vector => IVec}
  import scala.io.Source
  import java.io.{File, PrintWriter, OutputStreamWriter}
  import Types._
  import SpnExamples._
  import Mll._
  import Pmmh.runPmmh

  // This is how the data was simulated. Only re-run this if you _really_ want new data.
  def simData(): Unit = {
    import Sim._
    val ts = simTs(DenseVector(10, 0, 0, 0, 0), 0.0, 500.0, 10.0, stepAr(arparam))
    plotTs(ts)
    val s = new PrintWriter(new File("AR-perfect.txt"))
    s.write(toCsv(ts))
    s.close
    val nts = ts.map(r => (r._1, r._2.map(_ * 1.0) + DenseVector(Gaussian(0.0, 10.0).sample(5).toArray)))
    plotTs(nts)
    val s2 = new PrintWriter(new File("AR-noise10.txt"))
    s2.write(toCsv(nts))
    s2.close
  }

  def simPrior(th: ArParameter)(n: Int, t: Time): IVec[IntState] = {
    // Assume known initial state for now...
    val v = DenseVector(10, 0, 0, 0, 0)
    IVec.fill(n)(v)
  }

  def obsLik(th: ArParameter)(s: IntState, o: DoubleState): LogLik = {
    // Hard code a noise standard deviation of 10
    // Observing P and P2 only
    Gaussian(s(3).toDouble, 10.0).logPdf(o(3)) +
      Gaussian(s(4).toDouble, 10.0).logPdf(o(4))
  }

  def obsLikFull(th: ArParameter)(s: IntState, o: DoubleState): LogLik = {
    // Hard code a noise standard deviation of 10
    // Observing all species
    val ll = (s.toArray.toList zip o.toArray.toList) map { t =>
      Gaussian(t._1.toDouble, 10.0).logPdf(t._2)
    }
    ll.foldLeft(0.0)(_ + _)
  }

  def runModel(its: Int): Unit = {
    val rawData = Source.fromFile("AR-noise10.txt").getLines.toList
    val data = rawData.map(_.split(",")).map(
      r => (r.head.toDouble, r.tail.map(_.toDouble))
    ).map(
        t => (t._1, DenseVector(t._2.toArray))
      )
    val mll = pfMllP(240, simPrior, 0.0, stepAr, obsLik, data)
    val s = new PrintWriter(new File("AR-Pmmh1k-240.csv"))
    // val s=new OutputStreamWriter(System.out)
    s.write((0 until 8).map(_.toString).map("c" + _).mkString(",") + "\n")
    val pmmhOutput = runPmmh(s, its, arparam, mll)
    s.close
  }

  def main(args: Array[String]): Unit = {
    // simData() // will re-generate a new simulated dataset!!! Only run if _sure_!
    runModel(1000)
  }

}

/* eof */
