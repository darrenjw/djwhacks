package bayeskit

class LvState(preyx: Int, predx: Int) extends State {
  val prey = preyx
  val pred = predx
  override def toString = "" + prey + "," + pred
}
object LvState {
  def apply(prey: Int, pred: Int) = new LvState(prey, pred)
}

class LvParameter(th0x: Double, th1x: Double, th2x: Double) extends Parameter {
  val th0 = th0x
  val th1 = th1x
  val th2 = th2x
  override def toString = "" + th0 + "," + th1 + "," + th2
}
object LvParameter {
  def apply(th0: Double, th1: Double, th2: Double) = new LvParameter(th0, th1, th2)
}

class LvObservation(obsx: Double) extends Observation {
  val obs = obsx
}
object LvObservation {
  def apply(obs: Double) = new LvObservation(obs)
}

object lvsim {

  import scala.math.{exp,log}
  import breeze.stats.distributions._
  import scala.annotation.tailrec
  import scala.io.Source
  import java.io.{ File, PrintWriter, OutputStreamWriter }
  import sim._
  import pmmh._
  import pfilter._

  def simPrior(n: Int, t: Time, th: Parameter): Vector[LvState] = {
    val prey = new Poisson(50.0).sample(n).toVector
    val predator = new Poisson(100.0).sample(n).toVector
    prey.zip(predator) map { x => LvState(x._1, x._2) }
  }

  def obsLik(s: LvState, o: LvObservation, th: Parameter): LogLik = {
    log(new Gaussian(s.prey, 10.0).pdf(o.obs))
  }

  def runModel(its: Int) = {
    val rawData = Source.fromFile("LVpreyNoise10.txt").getLines
    val data = ((0 to 30 by 2).toList zip rawData.toList) map { x => (x._1.toDouble, LvObservation(x._2.toDouble)) }
    val mll = pfPropPar(100, simPrior, 0.0, stepLV, obsLik, data)
    val s = new PrintWriter(new File("mcmc-out.csv"))
    // val s=new OutputStreamWriter(System.out)
    s.write("th1,th2,th3,")
    s.write(((0 to 30 by 2) map { n => "x" + n + ",y" + n }).mkString(",") + "\n")
    val pmmhOutput = runPmmhPath(s, its, LvParameter(1.0, 0.005, 0.6), mll, peturb)
    s.close
  }

  @tailrec def stepLV(x: LvState, t0: Time, dt: Time, th: LvParameter): LvState = {
    if (dt <= 0.0) x else {
      val h = (th.th0 * x.prey, th.th1 * x.pred * x.prey, th.th2 * x.pred)
      val h0 = h._1 + h._2 + h._3
      val t = if (h0 < 1e-10 || x.prey > 1e6) 1e99 else new Exponential(h0).draw
      if (t > dt) x else {
        val u = new Uniform(0, 1).draw // use simpler function!
        if (u < h._1 / h0) stepLV(LvState(x.prey + 1, x.pred), t0 + t, dt - t, th)
        else {
          if (u < (h._1 + h._2) / h0) stepLV(LvState(x.prey - 1, x.pred + 1), t0 + t, dt - t, th)
          else stepLV(LvState(x.prey, x.pred - 1), t0 + t, dt - t, th)
        }
      }
    }
  }

  def peturb(th: LvParameter): LvParameter = {
    LvParameter(th.th0 * exp(Gaussian(0, 0.01).draw), th.th1 * exp(Gaussian(0, 0.01).draw), th.th2 * exp(Gaussian(0, 0.01).draw))
  }

}