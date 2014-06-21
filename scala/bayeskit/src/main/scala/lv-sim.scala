package bayeskit

// concrete State for the LV model
class LvState(preyx: Int, predx: Int) extends State {
  val prey = preyx
  val pred = predx
  override def toString = "" + prey + "," + pred
}
// companion object
object LvState{
  def apply(prey: Int, pred: Int) = new LvState(prey,pred)
}

object lvsim {

  import breeze.stats.distributions._
  import scala.annotation.tailrec

  import sim._

  @tailrec def stepLV(x: LvState, t0: Time, dt: Time, th: Parameter): LvState = {
    if (dt <= 0.0) x else {
      val h = (th(0) * x.prey, th(1) * x.pred * x.prey, th(2) * x.pred)
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

}