/*
LvAbc.scala

Simple example for doing inference for the Lotka Volterra model using ABC methods

 */

package smfsb

object LvAbc {

  import scala.io.Source
  import breeze.linalg.DenseVector
  import breeze.stats.distributions.Uniform
  import java.io.{File, PrintWriter, OutputStreamWriter}
  import SpnExamples._
  import Types._
  import Step.pts
  import Abc._
  import Sim.simTs

  val rawData = Source.fromFile("LVpreyNoise10.txt").getLines.toList.map(_.toDouble)

  def lvModel(th: LvParameter): Ts[IntState] = simTs(DenseVector(50, 100), 0.0, 30.0, 2.0, stepLvPts(th))

  def lvDist(simd: Ts[IntState]): Double = {
    val prey = simd map { _._2.copy(0).toDouble }
    val diffs = (prey zip rawData) map { r => r._1 - r._2 }
    diffs.reduce((x, y) => math.sqrt(x * x + y * y))
  }

  def simPrior: LvParameter = {
    val th0 = math.exp(new Uniform(-6.0, 2.0).draw)
    val th1 = math.exp(new Uniform(-6.0, 2.0).draw)
    val th2 = math.exp(new Uniform(-6.0, 2.0).draw)
    LvParameter(th0, th1, th2)
  }

  def simPrior(n: Int): Vector[LvParameter] = {
    (0 until n).toVector map { x => simPrior }
  }

  val abcDist = abcDistance(lvModel, lvDist) _

  def pilotRun(n: Int): Double = {
    val abcSample = simPrior(n) // .par
    val dist = abcSample map { p => abcDist(p) }
    val sorted = dist.sorted
    val cut = sorted(n / 200)
    cut
  }

  def runModel(n: Int): Unit = {
    println("starting pilot")
    val cutoff = pilotRun(50000)
    println("finished pilot. starting prior sim")
    val priorSample = simPrior(n) // .par
    println("finished prior sim. starting main forward sim")
    val dist = priorSample map { p => abcDist(p) }
    println("finished main sim. tidying up")
    val abcSample = (priorSample zip dist) filter (_._2 < cutoff)
    println(abcSample.length)
    val s = new PrintWriter(new File("LVPN10-Abc1m.csv"))
    // val s=new OutputStreamWriter(System.out)
    s.write("th0,th1,th2,d\n")
    abcSample map {t => s.write(t._1.toCsv+","+t._2+"\n")}
    s.close

  }

  def main(args: Array[String]): Unit = {
    runModel(1000000)
  }

}

/* eof */

