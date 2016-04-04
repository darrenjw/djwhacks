/*
ArAbcSs.scala

Simple example for doing inference for the AutoReg model using ABC methods with
summary statistics

 */

package smfsb

object ArAbcSs {

  import scala.io.Source
  import breeze.linalg.DenseVector
  import breeze.stats.distributions.{Uniform,Gaussian}
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

  // Assume known initial state for now...
  def arModel(th: ArParameter): Ts[DoubleState] = {
    val step = stepAr
    // val step = Step.pts(ar, 0.001)
    val ts=simTs(
      DenseVector(10, 0, 0, 0, 0),
      0.0, 500.0, 10.0, step(th)
    )
    val nts = ts.map(r => (r._1, r._2.map(_ * 1.0) +
      DenseVector(Gaussian(0.0, 10.0).sample(5).toArray)))
    nts
  }

  // raw summary stats, prior to rescaling
  def rawSs(simd: Ts[DoubleState]): DenseVector[Double] = {
    val sp = simd.map(_._2(3))
    val sp2 = simd.map(_._2(4))
    import breeze.stats._
    val mav=meanAndVariance(sp)
    val mav2=meanAndVariance(sp2)
    // TODO: Add in other summary stats!!!!! ***********
    DenseVector(mav.mean,math.log(mav.variance),mav2.mean,math.log(mav2.variance))
    }

  // given a vector of SDs, returns a function for scaled summary stats
  def getSs(sds: DenseVector[Double]): Ts[DoubleState] => DenseVector[Double] = {
    ts => rawSs(ts) :/ sds
  }

  // given a function for scaled summary stats, returns a function for euclidean distance
  def getMetric(ss: Ts[DoubleState] => DenseVector[Double]): Ts[DoubleState] => Double = {
    import breeze.linalg._
    val ss0 = ss(data)
    ts => {
      val d=ss0 - ss(ts)
      val ds=d :* d
      sum(ds)
      }
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

  def simPrior(n: Int): Vector[ArParameter] = {
    (0 until n).toVector map { x => simPrior }
  }

  def pilotRun(n: Int): (Ts[DoubleState]=>Double,Double) = {
    import breeze.stats._
    println("starting pilot")
    val abcSample = simPrior(n).par
    val dataSets= abcSample map (p => arModel(p))
    val rss = dataSets map { ts => rawSs(ts) }
    val d=rss(0).length
    val vecs = (0 until d).map(i => rss.map(p=>p(i)))
    val sds=DenseVector(vecs.map(v=>stddev(v.toArray)).toArray)
    println("sds: "+sds)
    val ss=getSs(sds)
    val metric=getMetric(ss)
    val dist=dataSets map (ts => metric(ts))
    val sorted = dist.toVector.sorted
    val cut = sorted(n / 200)
    println("finished pilot")
    (metric,cut)
  }

  def runModel(n: Int): Unit = {
    val (metric,cutoff) = pilotRun(10000)
    println("cutoff is " + cutoff)
    val distance=abcDistance(arModel,metric) _
    println("starting prior sim")
    val priorSample = simPrior(n).par
    println("finished prior sim. starting main forward sim")
    val dist = priorSample map { p => distance(p) }
    println("finished main sim. tidying up")
    val abcSample = (priorSample zip dist) filter (_._2 < cutoff)
    println("final sample size: "+abcSample.length)
    val s = new PrintWriter(new File("AR-AbcSs10k.csv"))
    // val s=new OutputStreamWriter(System.out)
    s.write((0 until 8).map(_.toString).map("c" + _).mkString(",") + ",distance\n")
    abcSample map { t => s.write(t._1.toCsv + "," + t._2 + "\n") }
    s.close

  }

  def main(args: Array[String]): Unit = {
    runModel(10000)
  }

}

/* eof */

