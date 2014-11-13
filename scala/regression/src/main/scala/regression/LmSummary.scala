package regression

import breeze.linalg._
import breeze.stats._
import breeze.stats.DescriptiveStats._
import org.saddle._
import scala.math.sqrt
import breeze.numerics

class LmSummary(m: Lm) {

  // TODO: Map over cols for multiple outputs

  val five = fiveNumber(m.residuals(::, 0))
  val df = m.y.rows - m.names.length
  val rse = sqrt(sum(m.residuals(::, 0) :^ 2.0) / df)
  val ri=inv(m.r)
  val xtxi=ri*(ri.t)
  val se=numerics.sqrt(diag(xtxi))*rse

  override def toString = {
    m.toString + five + Series(Vec(rse, df), Index("RSE", "df")) + se
  }

  def fiveNumber(v: DenseVector[Double]): Series[String, Double] = {
    val a = v.toArray
    val f = Array(0.0, 0.25, 0.5, 0.75, 1.0) map { percentile(a, _) }
    Series(Vec(f), Index("Min", "LQ", "Median", "UQ", "Max"))
  }

}