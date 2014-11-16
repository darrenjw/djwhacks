package regression

import breeze.linalg._
import breeze.stats._
import breeze.stats.DescriptiveStats._
import org.saddle._
import scala.math.sqrt
import breeze.numerics
import breeze.stats.distributions.{ Gaussian, StudentsT }
import FrameUtils._

class LmSummary(m: Lm) {

  // TODO: Map over cols for multiple outputs!
  // Currently assuming a single output!

  val five = fiveNumber(m.residuals(::, 0))
  val n = m.y.rows
  val pp = m.names.length
  val df = n - pp
  val rss = sum(m.residuals(::, 0) :^ 2.0)
  val rse = sqrt(rss / df)
  val ri = inv(m.r)
  val xtxi = ri * (ri.t)
  val se = numerics.sqrt(diag(xtxi)) * rse
  val seF = Frame(Vec(se.toArray), m.coeffFrame.rowIx, Index("SE"))
  val t = m.coefficients(::, 0) / se
  val tF = Frame(Vec(t.toArray), m.coeffFrame.rowIx, Index("t-val"))
  //val p=t.map{1.0-StudentsT(df).cdf(_)}.map{_*2} // .cdf missing... filed an issue - no incomplete beta function
  val p = t.map { 1.0 - Gaussian(0.0, 1.0).cdf(_) }.map { _ * 2 } // TODO: Gaussian approximate p-value for now... 
  val pF = Frame(Vec(p.toArray), m.coeffFrame.rowIx, Index("p-val"))
  val coeff = joinFrames(List(m.coeffFrame, seF, tF, pF))
  val ybar = mean(m.y(::, 0))
  val ymyb = m.y(::, 0) - ybar
  val ssy = sum(ymyb :^ 2.0)
  val rSquared = (ssy - rss) / ssy
  val adjRs = 1.0 - ((n - 1.0) / (n - pp)) * (1 - rSquared)
  val k = pp - 1
  val f = (ssy - rss) / k / (rss / df) // p-val is F on k and df under H0. No F in Breeze, and no incomplete beta function...

  override def toString = {
    "Residuals:\n" + five +
      "Coefficients:\n" + coeff +
      "Model statistics:\n" + Series(Vec(rss, rse, df, rSquared, adjRs, f), Index("RSS", "RSE", "df", "R-squared", "Adjusted R-sq", "F-stat"))
  }

  def fiveNumber(v: DenseVector[Double]): Series[String, Double] = {
    val a = v.toArray
    val f = Array(0.0, 0.25, 0.5, 0.75, 1.0) map { percentile(a, _) }
    Series(Vec(f), Index("Min", "LQ", "Median", "UQ", "Max"))
  }

}