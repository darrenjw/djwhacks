/*
ArAbc.scala

Simple example for doing inference for the AutoReg model using ABC methods

 */

package smfsb

object ArAbc {

  import scala.io.Source
  import breeze.linalg.DenseVector
  import breeze.stats.distributions.Uniform
  import java.io.{File, PrintWriter, OutputStreamWriter}
  import SpnExamples._
  import Types._
  import Step.pts
  import Abc._
  import Sim.simTs

  val rawData = Source.fromFile("AR-noise10.txt").getLines.toList
  val data = rawData.map(_.split(",")).map(
    r => (r.head.toDouble, r.tail.map(_.toDouble))
  ).map(
      t => (t._1, DenseVector(t._2.toArray))
    )
  val p = data.map(_._2(3))
  val p2 = data.map(_._2(4))

  // Assume known initial state for now...
  def arModel(th: ArParameter): Ts[IntState] = {
    val step = stepAr
    // val step = Step.pts(ar, 0.001)
    simTs(
      DenseVector(10, 0, 0, 0, 0),
      0.0, 500.0, 10.0, step(th)
    )
  }

  def arDist(simd: Ts[IntState]): Double = {
    // Matching on P and P2 only...
    val sp = simd.map(_._2(3))
    val sp2 = simd.map(_._2(4))
    val dp = (sp zip p) map { r => r._1 - r._2 }
    val dps = dp.reduce((x, y) => math.sqrt(x * x + y * y))
    val dp2 = (sp2 zip p2) map { r => r._1 - r._2 }
    val dps2 = dp2.reduce((x, y) => math.sqrt(x * x + y * y))
    dps + dps2
  }

  def simPrior: ArParameter = {
    val c = DenseVector(
      1.0,
      10.0,
      0.1,
      math.exp(Uniform(-1, 4).draw),
      1.0,
      math.exp(Uniform(-2, 3).draw),
      math.exp(Uniform(-3, 2).draw),
      math.exp(Uniform(-6, -1).draw)
    )
    ArParameter(c)
  }

  def simPriorFull: ArParameter = {
    val c = DenseVector(
      Uniform(-2, 3).draw,
      Uniform(-1, 4).draw,
      Uniform(-6, -1).draw,
      Uniform(-1, 4).draw,
      Uniform(-2, 3).draw,
      Uniform(-2, 3).draw,
      Uniform(-3, 2).draw,
      Uniform(-6, -1).draw
    ).map(math.exp(_))
    ArParameter(c)
  }

  def simPrior(n: Int): Vector[ArParameter] = {
    (0 until n).toVector map { x => simPrior }
  }

  val abcDist = abcDistance(arModel, arDist) _

  def pilotRun(n: Int): Double = {
    val abcSample = simPrior(n).par
    val dist = abcSample map { p => abcDist(p) }
    val sorted = dist.toVector.sorted
    val cut = sorted(n / 100)
    cut
  }

  def runModel(n: Int): Unit = {
    println("starting pilot")
    val cutoff = pilotRun(10000)
    println("cutoff is " + cutoff)
    println("finished pilot. starting prior sim")
    val priorSample = simPrior(n).par
    println("finished prior sim. starting main forward sim")
    val dist = priorSample map { p => abcDist(p) }
    println("finished main sim. tidying up")
    val abcSample = (priorSample zip dist) filter (_._2 < cutoff)
    println("final sample size: "+abcSample.length)
    val s = new PrintWriter(new File("AR-Abc100k.csv"))
    // val s=new OutputStreamWriter(System.out)
    s.write((0 until 8).map(_.toString).map("c" + _).mkString(",") + ",distance\n")
    abcSample map { t => s.write(t._1.toCsv + "," + t._2 + "\n") }
    s.close

  }

  def main(args: Array[String]): Unit = {
    runModel(100000)
  }

}

/* eof */

