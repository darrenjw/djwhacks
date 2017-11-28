/*
predator-prey.scala
Main runner script
*/

object PredatorPrey {

  import java.io.{ File, PrintWriter }
  import breeze.linalg._
  import breeze.plot._
  import breeze.stats.distributions.{ Gaussian, Uniform }
  import montescala.TypeClasses._
  import scala.collection.parallel.immutable.ParVector
  import scala.collection.immutable.{ Vector => IVec }
  import math.{ exp, log }

  type DMD = DenseMatrix[Double]
  type DVD = DenseVector[Double]

  def readData(fileName: String = "raw-data.csv"): DMD = {
    println("Reading data from disk")
    val raw = csvread(new File(fileName), skipLines = 1)
    println(raw.rows.toString + " rows")
    println(raw.cols.toString + " cols (should be 4)")
    raw
  }

  def plotData(raw: DMD): Figure = {
    println("Plotting data")
    val day = raw(::, 1)
    val virus = raw(::, 2)
    val aob = raw(::, 3)
    val fig = Figure("Raw data")
    fig.width = 800
    fig.height = 600
    val p1 = fig.subplot(2, 1, 0)
    p1 += plot(day, aob)
    p1.title = "AOB"
    p1.xlabel = "Day"
    p1.ylabel = "# AOB / mL"
    val p2 = fig.subplot(2, 1, 1)
    p2 += plot(day, virus, colorcode = "red")
    p2.title = "Virus"
    p2.xlabel = "Day"
    p2.ylabel = "# Virus / mL"
    fig
  }

  case class LvParam(
    mu: Double, phi: Double, delta: Double, m: Double,
    vx: Double, vv: Double,
    nvx: Double, nvv: Double) {
    def toCsv: String = s"$mu,$phi,$delta,$m,$vx,$vv,$nvx,$nvv"
  }
  implicit val lvParameter = new Parameter[LvParam] {}

  case class LvState(x: Double, v: Double)
  implicit val lvState = new State[LvState] {}

  case class LvObs(x: Double, v: Double)
  implicit val lvObs = new Observation[LvObs] {}

  @annotation.tailrec
  def stepLV(dt: Double)(
    p: LvParam)(
      s: LvState, deltat: Double): LvState = {
    if (deltat <= 0.0) s else {
      val newX = s.x + (p.mu * s.x - p.phi * s.x * s.v) * dt +
        Gaussian(0.0, math.sqrt(p.vx * s.x * dt)).draw
      val newV = s.v + (p.delta * s.x * s.v - p.m * s.v) * dt +
        Gaussian(0.0, math.sqrt(p.vv * s.v * dt)).draw
      stepLV(dt)(p)(LvState(math.max(1000.0, newX), math.max(1000.0, newV)), deltat - dt)
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
    val p1 = fig.subplot(2, 1, 0)
    p1 += plot(linspace(1.0, n, n), aob)
    p1.title = "AOB"
    p1.xlabel = "Day"
    p1.ylabel = "# AOB / mL"
    val p2 = fig.subplot(2, 1, 1)
    p2 += plot(linspace(1.0, n, n), virus, colorcode = "red")
    p2.title = "Virus"
    p2.xlabel = "Day"
    p2.ylabel = "# Virus / mL"
    fig
  }

  val s0 = LvState(1692200.0, 3406695410.0) // initial observation

  def simPrior(n: Int)(p: LvParam): ParVector[LvState] = {
    IVec.fill(n)(LvState(
      math.max(1.0, Gaussian(s0.x, 100000.0).draw),
      math.max(1.0, Gaussian(s0.v, 100000000.0).draw))).par
  }

  def dataLik(p: LvParam)(s: LvState, o: LvObs): LogLik = {
    Gaussian(s.x, math.sqrt(p.nvx)).logPdf(o.x) +
      Gaussian(s.v, math.sqrt(p.nvv)).logPdf(o.v)
  }

  val minParam = 1.0e-15

  // all parameters constrained to be positive
  def isValidP(p: LvParam): Boolean = (
    (p.mu > minParam) && (p.phi > minParam) && (p.delta > minParam) && (p.m > minParam) && (p.vx > minParam) && (p.vv > minParam) && (p.nvx > minParam) && (p.nvv > minParam))

  // noise parameters (only) constrained to be positive
  def isValid(p: LvParam): Boolean = (
    (p.vx > minParam) && (p.vv > minParam) && (p.nvx > minParam) && (p.nvv > minParam))

  // all parameters constrained to be positive
  def genPropP(p: LvParam, tune: Double): LvParam =
    LvParam(
      p.mu * exp(Gaussian(0.0, tune).draw),
      p.phi * exp(Gaussian(0.0, tune).draw),
      p.delta * exp(Gaussian(0.0, tune).draw),
      p.m * exp(Gaussian(0.0, tune).draw),
      p.vx * exp(Gaussian(0.0, tune).draw),
      p.vv * exp(Gaussian(0.0, tune).draw),
      p.nvx * exp(Gaussian(0.0, tune).draw),
      p.nvv * exp(Gaussian(0.0, tune).draw))

  // noise parameters (only) constrained to be positive
  def genProp(p: LvParam, tune: Double): LvParam =
    LvParam(
      p.mu + Gaussian(0.0, tune).draw,
      p.phi + Gaussian(0.0, tune).draw,
      p.delta + Gaussian(0.0, tune).draw,
      p.m + Gaussian(0.0, 4.0 * tune).draw, // boost variance of m proposal
      p.vx * exp(Gaussian(0.0, 0.1).draw),
      p.vv * exp(Gaussian(0.0, 0.1).draw),
      p.nvx * exp(Gaussian(0.0, 0.1).draw),
      p.nvv * exp(Gaussian(0.0, 0.1).draw))

  def nextIter(mll: LvParam => LogLik, tune: Double)(tup: (LvParam, LogLik)): (LvParam, LogLik) = {
    val (p, ll) = tup
    val prop = genProp(p, tune)
    val pll = mll(prop)
    val logA = pll - ll
    if (isValid(prop) && (log(Uniform(0.0, 1.0).draw) < logA))
      (prop, pll) else tup
  }

  def mllVar(mll: LvParam => LogLik, n: Int, p: LvParam): Double = {
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
      //val p0 = LvParam(2.664662007004367E-11,1.0407442929184285E-12,1.0216394661768682E-10,5.8971462269015E-13,5787334.243841605,3.2687200652293317E7,3.373932185750761E12,8.6864383170127936E16)
      val p0 = LvParam(3.718792068332797E-12, 1.430788978108719E-11, -1.3306417869347798E-10, 5.560371094820542E-9, 7320141.707961543, 5.978856231765036E7, 1.2828110376132417E12, 3.2545757491941356E16)
      println(s"its: $its, N: $N, thin: $thin, tune: $tune")
      val raw = readData()
      //plotData(raw)
      //plotTs(s0,100)(stepLV(dt)(p0)(_,timeStep))
      val data = (0 until raw.rows) map (r => LvObs(raw(r, 3), raw(r, 2)))
      val mll = pfMll(
        simPrior(N),
        (p: LvParam) => (s: LvState) => stepLV(dt)(p)(s, timeStep),
        dataLik,
        (zc: ParVector[(Double, LvState)], srw: Double, l: Int) =>
          resampleSys(zc, srw, l),
        data)
      //mllVar(mll,100,p0)
      import Thinnable.ops._
      val pmmh = Stream.iterate((p0, Double.MinValue))(nextIter(mll, tune))
      println("Running PMMH MCMC now...")
      val s = new PrintWriter(new File("LvPmmh.csv"))
      s.write("mu,phi,delta,m,vx,vv,nvx,nvv,ll\n")
      pmmh.thin(thin).take(its).foreach(tup => {
        print(".")
        s.write(tup._1.toCsv + "," + tup._2 + "\n")
      })
      println("\nMCMC Done.")
      s.close
      println("Bye...")
    }
  }

}
