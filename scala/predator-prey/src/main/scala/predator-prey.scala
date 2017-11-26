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

  type DMD = DenseMatrix[Double]
  type DVD = DenseVector[Double]

  def readData(fileName: String = "raw-data.csv"): DMD = {
    println("Reading data from disk")
    val raw = csvread(new File(fileName),skipLines=1)
    println(raw.rows.toString+" rows")
    println(raw.cols.toString+" cols (should be 4)")
    raw
  }

  def plotData(raw: DMD): Figure = {
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

  case class LvState(x: Double, v: Double)
  implicit val lvState = new State[LvState] {}

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
      val newX = s.x + (p.mu*s.x - p.phi*s.x*s.v)*dt +
      Gaussian(0.0,math.sqrt(p.vx*s.x*dt)).draw
      val newV = s.v + (p.delta*s.x*s.v - p.m*s.v)*dt +
      Gaussian(0.0,math.sqrt(p.vv*s.v*dt)).draw
      stepLV(dt)(p)(LvState(math.max(1000.0,newX),math.max(1000.0,newV)),deltat-dt)
    }
  }

  def plotTs(s0: LvState, n: Int)(stepFun: LvState => LvState): Figure = {
    println("Plotting simulated time series")
    val sim = Stream.iterate(s0)(stepFun).take(n).toArray
    val aob = DenseVector(sim map (_.x))
    val virus = DenseVector(sim map (_.v))
    val fig = Figure("Simulated data")
    fig.width = 800
    fig.height = 600
    val p1 = fig.subplot(2,1,0)
    p1 += plot(linspace(1.0,n,n),aob)
    p1.title = "AOB"
    p1.xlabel = "Day"
    p1.ylabel = "# AOB / mL"
    val p2 = fig.subplot(2,1,1)
    p2 += plot(linspace(1.0,n,n),virus,colorcode="red")
    p2.title = "Virus"
    p2.xlabel = "Day"
    p2.ylabel = "# Virus / mL"
    fig
  }

  val s0 = LvState(1692200.0,3406695410.0) // initial observation

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

  val minParam = 1.0e-15

  def isValid(p: LvParam): Boolean = (
    (p.mu > minParam) && (p.phi > minParam) && (p.delta > minParam) && (p.m > minParam) && (p.vx > minParam) && (p.vv > minParam) && (p.nvx > minParam) && (p.nvv > minParam)
  )

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
    if (isValid(prop) && (math.log(Uniform(0.0,1.0).draw) < logA)) {
      //println("Accept")
      (prop, pll)
    }
    else {
      //println("Reject")
      tup
    }
  }

  def mllVar(mll: LvParam => LogLik,n: Int, p: LvParam): Double = {
    println(s"Testing mll variance with $n evaluations")
    val x = DenseVector.fill(n)(mll(p))
    import breeze.stats._
    val tup = meanAndVariance(x)
    val m = tup.mean
    val v = tup.variance
    println(s"Mean is $m")
    println(s"Variance is $v")
    v
  }

  import montescala.BPFilter._

  def main(args: Array[String]): Unit = {
    println("LV PMMH")
    if (args.length != 4) {
      println("From SBT: run <its> <parts> <thin> <tune>")
      println("eg. run 10000 2000 10 0.2")
    } else {
      val its = args(0).toInt // Number of MCMC iterations (AFTER thinning)
      val N = args(1).toInt // Number of particles for BPFilter
      val thin = args(2).toInt // MCMC thinning
      val tune = args(3).toDouble // M-H tuning parameter
      val dt = 0.1 //  for Euler Maruyama
      val timeStep = 1.0 // inter-observation time
      //val p0 = LvParam(1.0,1.0e-10,1.0e-5,1.0,100.0,1000.0,1000000000.0,10000000000.0)
      val p0 = LvParam(2.664662007004367E-11,1.0407442929184285E-12,1.0216394661768682E-10,5.8971462269015E-13,5787334.243841605,3.2687200652293317E7,3.373932185750761E12,8.6864383170127936E16)
      println(s"its: $its, N: $N, thin: $thin, tune: $tune")
      val raw = readData()
      //plotData(raw)
      //plotTs(s0,100)(stepLV(dt)(p0)(_,timeStep))
      val data = (0 until raw.rows) map (r => LvObs(raw(r,3),raw(r,2)))
      val mll = pfMll(
      simPrior(N),
        (p: LvParam) => (s: LvState) => stepLV(dt)(p)(s,timeStep),
      dataLik,
      (zc: ParVector[(Double, LvState)], srw: Double, l: Int) =>
      resampleSys(zc,srw,l),
        data)
      //mllVar(mll,100,p0)
      import Thinnable.ops._
      val pmmh = Stream.iterate((p0,Double.MinValue))(nextIter(mll,tune))
      println("Running PMMH MCMC now...")
      val s = new PrintWriter(new File("LvPmmh.csv"))
      s.write("mu,phi,delta,m,vx,vv,nvx,nvv,ll\n")
      pmmh.thin(thin).take(its).foreach{tup => {
        print(".")
        s.write(tup._1.toCsv+","+tup._2+"\n")
      }}
      println("\nMCMC Done.")
      s.close
      println("Bye...")
    }
  }

}
