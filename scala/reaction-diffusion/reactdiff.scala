/*
reactdiff.scala
2d reaction diffusion of the Lotka Volterra predator prey system
Discrete stochastic reaction diffusion master equation model
Simulated with the next subvolume method

 */

object ReactDiff2d {

  import annotation.tailrec

  import breeze.linalg._
  import breeze.math._
  import breeze.numerics._
  import breeze.stats.distributions.{Uniform, Exponential, Multinomial}

  import java.awt.image.BufferedImage

  val D = 50
  val T = 120
  val dt = 0.25
  val th = Vector(1.0, 0.005, 0.6)
  val dc = 0.25

  val N = (T / dt).toInt
  val x = DenseMatrix.zeros[Int](D, D)
  x(D / 2, D / 2) = 60
  val y = DenseMatrix.zeros[Int](D, D)
  y(D / 2, D / 2) = 60

  def mvLeft(m: Matrix[Int], i: Int, j: Int): (Int, Int) = {
    m(i, j) -= 1
    val jj = if (j > 0) j - 1 else m.cols - 1
    m(i, jj) += 1
    (i, jj)
  }

  def mvRight(m: Matrix[Int], i: Int, j: Int): (Int, Int) = {
    m(i, j) -= 1
    val jj = if (j < m.cols - 1) j + 1 else 0
    m(i, jj) += 1
    (i, jj)
  }

  def mvUp(m: Matrix[Int], i: Int, j: Int): (Int, Int) = {
    m(i, j) -= 1
    val ii = if (i > 0) i - 1 else m.rows - 1
    m(ii, j) += 1
    (ii, j)
  }

  def mvDown(m: Matrix[Int], i: Int, j: Int): (Int, Int) = {
    m(i, j) -= 1
    val ii = if (i < m.rows - 1) i + 1 else 0
    m(ii, j) += 1
    (ii, j)
  }

  def diffuse(x: Matrix[Int], y: Matrix[Int], hd: DenseMatrix[Double]): List[(Int, Int)] = {
    val r = Multinomial(hd.toDenseVector).draw
    val i = r % D
    val j = r / D
    val u = Uniform(0.0, 1.0).draw
    val extra = if (u < 0.25) {
      if (Uniform(0.0, x(i, j) + y(i, j)).draw < x(i, j)) mvLeft(x, i, j) else mvLeft(y, i, j)
    } else if (u < 0.5) {
      if (Uniform(0.0, x(i, j) + y(i, j)).draw < x(i, j)) mvRight(x, i, j) else mvRight(y, i, j)
    } else if (u < 0.75) {
      if (Uniform(0.0, x(i, j) + y(i, j)).draw < x(i, j)) mvUp(x, i, j) else mvUp(y, i, j)
    } else {
      if (Uniform(0.0, x(i, j) + y(i, j)).draw < x(i, j)) mvDown(x, i, j) else mvDown(y, i, j)
    }
    List((i, j), extra)
  }

  def react(x: Matrix[Int], y: Matrix[Int], h: Vector[DenseMatrix[Double]], hr: DenseMatrix[Double]): List[(Int, Int)] = {
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
    List((i, j))
  }

  def recalc(pix: (Int, Int), x: DenseMatrix[Int], y: DenseMatrix[Int], h: Vector[DenseMatrix[Double]], hr: DenseMatrix[Double], hd: DenseMatrix[Double]): (Double, Double) = {
    val i = pix._1
    val j = pix._2
    val oldhr = h(0)(i, j) + h(1)(i, j) + h(2)(i, j)
    val oldhd = hd(i, j)
    h(0)(i, j) = th(0) * x(i, j)
    h(1)(i, j) = th(1) * x(i, j) * y(i, j)
    h(2)(i, j) = th(2) * y(i, j)
    val newhr = h(0)(i, j) + h(1)(i, j) + h(2)(i, j)
    hr(i, j) = newhr
    val newhd = 4.0 * dc * (x(i, j) + y(i, j))
    hd(i, j) = newhd
    (newhr - oldhr, newhd - oldhd)
  }

  def stepLV(x: DenseMatrix[Int], y: DenseMatrix[Int], dt: Double): (DenseMatrix[Int], DenseMatrix[Int]) = {
    val xc = x.copy
    val yc = y.copy
    val h = Vector[DenseMatrix[Double]](x map { _ * th(0) }, (x :* y) map { _ * th(1) }, y map { _ * th(2) })
    val hr = h(0) + h(1) + h(2)
    val hrs = sum(hr)
    val hd = ((x + y) map { _ * dc }) * 4.0
    val hds = sum(hd)
    @tailrec
    def go(x: DenseMatrix[Int], y: DenseMatrix[Int], dt: Double, hrs: Double, hds: Double): (DenseMatrix[Int], DenseMatrix[Int]) = {
      val h0 = hrs + hds
      val et = Exponential(h0).draw
      if (et > dt) (x, y) else {
        val touched = if (Uniform(0.0, h0).draw < hds) {
          diffuse(x, y, hd)
        } else {
          react(x, y, h, hr)
        }
        val deltah = (touched map { recalc(_, x, y, h, hr, hd) }).foldLeft((0.0, 0.0))((a, b) => (a._1 + b._1, a._2 + b._2))
        val newhrs = hrs + deltah._1
        val newhds = hds + deltah._2
        go(x, y, dt - et, newhrs, newhds)
      }
    }
    go(xc, yc, dt, hrs, hds)
  }

  def mkImage(x: DenseMatrix[Int], y: DenseMatrix[Int]): BufferedImage = {
    val canvas = new BufferedImage(D, D, BufferedImage.TYPE_INT_RGB)
    val wr = canvas.getRaster
    val mx = max(x)
    val my = max(y)
    for (i <- 0 until D) {
      for (j <- 0 until D) {
        wr.setSample(i, j, 2, 255 * x(i, j) / mx) // band 2 is blue
        wr.setSample(i, j, 0, 255 * y(i, j) / my) // band 0 is red
      }
    }
    canvas
  }

  def main(args: Array[String]): Unit = {
    println("Hello")
    @tailrec
    def go(x: DenseMatrix[Int], y: DenseMatrix[Int], left: Int): Unit = {
      if (left > 0) {
        print("" + left + " ")
        val i = N - left
        val next = stepLV(x, y, dt)
        val xt = next._1
        val yt = next._2
        val im = mkImage(xt, yt)
        javax.imageio.ImageIO.write(im, "png", new java.io.File(f"img$i%04d.png"))
        go(xt, yt, left - 1)
      }
    }
    go(x, y, N)
    println("Goodbye")
  }

}

// eof

