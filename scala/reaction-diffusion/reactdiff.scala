/*
reactdiff.scala
2d reaction diffusion of the Lotka Volterra predator prey system
Discrete stochastic reaction diffusion master equation model
Simulated with the next subvolume method

 */

object ReactDiff2d {

  import breeze.linalg._
  import breeze.math._
  import breeze.numerics._
  import breeze.stats.distributions.{Uniform, Exponential, Multinomial}

  import breeze.plot._

  val D = 50
  val T = 120
  val dt = 0.25
  val th = Vector(1.0, 0.005, 0.6)
  val dc = 0.25

  val N = T / dt
  val x = DenseMatrix.zeros[Int](D, D)
  x(D / 2, D / 2) = 60
  val y = DenseMatrix.zeros[Int](D, D)
  y(D / 2, D / 2) = 60

  def mvLeft(m: Matrix[Int], i: Int, j: Int): Unit = {
    m(i, j) -= 1
    if (j > 0) m(i, j - 1) += 1 else m(i, m.cols - 1) += 1
  }

  def mvRight(m: Matrix[Int], i: Int, j: Int): Unit = {
    m(i, j) -= 1
    if (j < m.cols - 1) m(i, j + 1) += 1 else m(i, 0) += 1
  }

  def mvUp(m: Matrix[Int], i: Int, j: Int): Unit = {
    m(i, j) -= 1
    if (i > 0) m(i - 1, j) += 1 else m(m.rows - 1, j) += 1
  }

  def mvDown(m: Matrix[Int], i: Int, j: Int): Unit = {
    m(i, j) -= 1
    if (i < m.rows - 1) m(i + 1, j) += 1 else m(0, j) += 1
  }

  def diffuse(x: Matrix[Int], y: Matrix[Int], hd: DenseMatrix[Double]): Unit = {
    val r = Multinomial(hd.toDenseVector).draw
    val i = r % D
    val j = r / D
    val u = Uniform(0.0, 1.0).draw
    if (u < 0.25) {
      if (Uniform(0.0, x(i, j) + y(i, j)).draw < x(i, j)) mvLeft(x, i, j) else mvLeft(y, i, j)
    } else if (u < 0.5) {
      if (Uniform(0.0, x(i, j) + y(i, j)).draw < x(i, j)) mvRight(x, i, j) else mvRight(y, i, j)
    } else if (u < 0.75) {
      if (Uniform(0.0, x(i, j) + y(i, j)).draw < x(i, j)) mvUp(x, i, j) else mvUp(y, i, j)
    } else {
      if (Uniform(0.0, x(i, j) + y(i, j)).draw < x(i, j)) mvDown(x, i, j) else mvDown(y, i, j)
    }
  }

  def react(x: Matrix[Int], y: Matrix[Int], h: Vector[DenseMatrix[Double]], hr: DenseMatrix[Double]): Unit = {
    val r = Multinomial(hr.toDenseVector).draw
    val i = r % D
    val j = r / D
    val u = Uniform(0.0, hr(i, j)).draw
    if (u < h(0)(i, j)) {
      x(i, j) += 1
    } else if (u < h(0)(i, j) + h(1)(i, j)) {
      x(i, j) -= 1
      y(i, j) += 1
    } else {
      y(i, j) -= 1
    }
  }

  // TODO: This should make a defensive copy at the top level
  @annotation.tailrec
  def stepLV(x: DenseMatrix[Int], y: DenseMatrix[Int], dt: Double): (DenseMatrix[Int], DenseMatrix[Int]) = {
    val h = Vector[DenseMatrix[Double]](x map { _ * th(0) }, (x :* y) map { _ * th(1) }, y map { _ * th(2) })
    val hr = h(0) + h(1) + h(2)
    val hrs = hr.sum
    val hd = ((x + y) map { _ * dc }) * 4.0
    val hds = hd.sum
    val h0 = hrs + hds
    val et = Exponential(h0).draw
    if (et > dt) (x, y) else {
      if (Uniform(0.0, h0).draw < hds) {
        diffuse(x, y, hd)
        stepLV(x, y, dt - et)
      } else {
        react(x,y,h,hr)
        stepLV(x, y, dt - et)
      }
    }
  }

  def main(args: Array[String]): Unit = {
    println("Hello")
    val f = Figure()
    for (i <- 1 to T) {
      println(i)
      val (xt, yt) = stepLV(x, y, dt)
      f.clear()
      f.subplot(0) += image(xt map { _ * 1.0 })
      f.subplot(1, 2, 1) += image(yt map { _ * 1.0 })
      // f.saveas("plot.png")
    }
    println("Goodbye")
  }

}

// eof

