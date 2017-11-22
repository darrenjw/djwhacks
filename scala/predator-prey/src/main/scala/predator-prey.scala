/*
predator-prey.scala
Main runner script
*/

object PredatorPrey {

  import java.io.{File,PrintWriter}
  import breeze.linalg._
  import breeze.plot._
  import montescala.TypeClasses._
  import scala.collection.parallel.immutable.ParVector
  import scala.collection.immutable.{Vector => IVec}

  def readData(fileName: String = "raw-data.csv"): DenseMatrix[Double] = {
    println("Reading data from disk")
    val raw = csvread(new File(fileName),skipLines=1)
    println(raw.rows.toString+" rows")
    println(raw.cols.toString+" cols (should be 4)")
    raw
  }

  def plotData(raw: DenseMatrix[Double]): Figure = {
    println("Plotting data")
    val day = raw(::,1)
    val virus = raw(::,2)
    val aob = raw(::,3)
    val fig = Figure("Raw data")
    fig.width = 800
    fig.height = 600
    val p1 = fig.subplot(2,1,0)
    p1 += plot(day,aob)
    p1.title = "AOB"
    p1.xlabel = "Day"
    p1.ylabel = "# AOB / mL"
    val p2 = fig.subplot(2,1,1)
    p2 += plot(day,virus,colorcode="red")
    p2.title = "Virus"
    p2.xlabel = "Day"
    p2.ylabel = "# Virus / mL"
    fig
  }

  case class LvParam(
    mu: Double, phi: Double, delta: Double, m: Double,
    vx: Double, vv: Double,
    nvx: Double, nvv: Double
  ) {
    def toCsv: String = s"$mu,$phi,$delta,$m,$vx,$vv,$nvx,$nvv"
  }
  implicit val lvParameter = new Parameter[LvParam] {}
  val p0 = LvParam(1.0,1.0,1.0,2.0,100.0,1000.0,1000000000.0,10000000000.0)

  case class LvState(x: Double, v: Double)
  implicit val lvState = new State[LvState] {}
  val s0 = LvState(1692200.0,3406695410.0) // initial observation

  case class LvObs(x: Double, v: Double)
  implicit val lvObs = new Observation[LvObs] {}

  import breeze.stats.distributions.Gaussian
  @annotation.tailrec
  def stepLV(dt: Double)(
    p: LvParam
  )(
    s: LvState, deltat: Double
  ): LvState = {
    if (deltat <= 0.0) s else {
      val newX = (p.mu*s.x - p.phi*s.x*s.v)*dt +
      Gaussian(0.0,math.sqrt(p.vx*s.x*dt)).draw
      val newV = (p.delta*s.x*s.v - p.m*s.v)*dt +
      Gaussian(0.0,math.sqrt(p.vv*s.v*dt)).draw
      stepLV(dt)(p)(LvState(math.max(0.0,newX),math.max(0.0,newV)),deltat-dt)
    }
  }

  def simPrior(n: Int)(p: LvParam): ParVector[LvState] = {
    IVec.fill(n)(LvState(
      Gaussian(s0.x,100000.0).draw,
      Gaussian(s0.v,10000000.0).draw
    )).par
  }

  def dataLik(p: LvParam)(s: LvState, o: LvObs): LogLik = {
    Gaussian(s.x,math.sqrt(p.nvx)).logPdf(o.x) +
    Gaussian(s.v,math.sqrt(p.nvv)).logPdf(o.v)
  }

  import math.exp
  import breeze.stats.distributions.Uniform
  def nextIter(mll: LvParam => LogLik,tune: Double)(tup: (LvParam, LogLik)): (LvParam, LogLik) = {
    val (p, ll) = tup
    val prop = LvParam(
      p.mu*exp(Gaussian(0.0,tune).draw),
      p.phi*exp(Gaussian(0.0,tune).draw),
      p.delta*exp(Gaussian(0.0,tune).draw),
      p.m*exp(Gaussian(0.0,tune).draw),
      p.vx*exp(Gaussian(0.0,tune).draw),
      p.vv*exp(Gaussian(0.0,tune).draw),
      p.nvx*exp(Gaussian(0.0,tune).draw),
      p.nvv*exp(Gaussian(0.0,tune).draw)
    )
    val pll = mll(prop)
    val logA = pll - ll
    //println(pll,ll,logA)
    if (math.log(Uniform(0.0,1.0).draw) < logA) {
      //println("Accept")
      (prop, pll)
    }
    else {
      //println("Reject")
      tup
    }
  }

  import montescala.BPFilter._

  def main(args: Array[String]): Unit = {
    println("LV PMMH")
    if (args.length != 3) {
      println("From SBT: run <its> <parts> <tune>")
      println("eg. run 10000 1000 0.1")
    } else {
      val its = args(0).toInt
      val N = args(1).toInt
      val tune = args(2).toDouble
      println(s"its: $its, N: $N, tune: $tune")
      val raw = readData()
      //plotData(raw)
      val data = (0 until raw.rows) map (r => LvObs(raw(r,3),raw(r,2)))
      val mll = pfMll(
      simPrior(N),
        (p: LvParam) => (s: LvState) => stepLV(0.05)(p)(s,1.0),
      dataLik,
      (zc: ParVector[(Double, LvState)], srw: Double, l: Int) =>
      resampleSys(zc,srw,l),
      data)
    val pmmh = Stream.iterate((p0,Double.MinValue))(nextIter(mll,tune))
    println("Running PMMH MCMC now...")
    val s = new PrintWriter(new File("LvPmmh.csv"))
    s.write("mu,phi,delta,m,vx,vv,nvx,nvv,ll\n")
    pmmh.take(its).foreach{tup => {
      print(".")
        s.write(tup._1.toCsv+","+tup._2+"\n")
    }}
    println("\nMCMC Done.")
      s.close
      println("Bye...")
    }
  }

}
