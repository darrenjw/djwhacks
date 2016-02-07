/*
reactdiffsg4.scala
2d reaction diffusion of the Lotka Volterra predator prey system
Spatial CLE approx

 */

object ReactDiff2dSGv4 {

  import annotation.tailrec
  // import Math.sqrt

  import breeze.linalg._
  import breeze.math._
  import breeze.numerics._
  import breeze.stats.distributions.Gaussian

  import java.awt.image.BufferedImage

  val D = 500
  val T = 100
  val dt = 0.1
  val th = Vector(1.0, 0.05, 0.6)
  val dc = 1.0

  val S = new DenseMatrix(2, 3, Array[Double](1, 0, -1, 1, 0, -1))

  val sdt=Math.sqrt(dt)
  println("sdt "+sdt)
  val sdc=Math.sqrt(dc)
  println("sdc "+sdc)

  val N = (T / dt).toInt
  val x = DenseMatrix.zeros[Double](D, D)
  x(100,100) = 60
  x(400,100) = 60
  x(250,400) = 20
  val y = DenseMatrix.zeros[Double](D, D)
  y(100, 100) = 20
  y(400, 100) = 20
  y(250, 400) = 10

  def up(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    DenseMatrix.vertcat(m(1 until m.rows, ::), m(0 to 0, ::))
  }

  def down(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    DenseMatrix.vertcat(m((m.rows - 1) to (m.rows - 1), ::), m(0 until (m.rows - 1), ::))
  }

  def left(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    DenseMatrix.horzcat(m(::, 1 until m.cols), m(::, 0 to 0))
  }

  def right(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    DenseMatrix.horzcat(m(::, (m.cols - 1) to (m.cols - 1)), m(::, 0 until (m.cols - 1)))
  }

  def laplace(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    up(m) + down(m) + left(m) + right(m) - (m * 4.0)
  }

  def rectify(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    m map { e => if (e > 0.0) e else 0.0 } // absorb
  }

  def sqrt(m: DenseMatrix[Double]): DenseMatrix[Double] = m map (Math.sqrt(_))

  def diffuse(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val dwt = new DenseMatrix(D, D, (Gaussian(0.0, sdt).sample(D * D)).toArray)
    val dwts = new DenseMatrix(D, D, (Gaussian(0.0, sdt).sample(D * D)).toArray)
    val md=laplace(m)*(dt*dc)
    val mh= ((sqrt(m + left(m)) :* dwt) - (sqrt(m + right(m)) :* right(dwt)))*sdc
    val mv=((sqrt(m + up(m)) :* dwts) - (sqrt(m + down(m)) :* down(dwts)))*sdc
    //println("sum mh is "+sum(mh))
    //println("sum mv is "+sum(mv))
    val mn=m+md +mh + mv
    //println("min mn is "+min(mn))
    rectify(mn)
  }

  def react(x: DenseMatrix[Double],y: DenseMatrix[Double]): (DenseMatrix[Double],DenseMatrix[Double]) = {
    val h1 = x * th(0)
    val h2 = (x :* y) * th(1)
    val h3 = y * th(2)
    val dw1t = new DenseMatrix(D, D, Gaussian(0.0, sdt).sample(D * D).toArray)
    val dw2t = new DenseMatrix(D, D, Gaussian(0.0, sdt).sample(D * D).toArray)
    val dw3t = new DenseMatrix(D, D, Gaussian(0.0, sdt).sample(D * D).toArray)
    val r1=(h1 * dt) + (sqrt(h1) :* dw1t)
    val r2=(h2 * dt) + (sqrt(h2) :* dw2t)
    val r3=(h3 * dt) + (sqrt(h3) :* dw3t)
    val dx = (r1 * S(0, 0)) + (r2 * S(0, 1)) + (r3 * S(0, 2))
    val dy = (r1 * S(1, 0)) + (r2 * S(1, 1)) + (r3 * S(1, 2))
    (rectify(x+dx), rectify(y+dy))

}

  def stepLV(x: DenseMatrix[Double], y: DenseMatrix[Double], dt: Double): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val xd = diffuse(x)
    val yd = diffuse(y)
    react(xd,yd)
    //(xd,yd)
  }

  def mkImage(x: DenseMatrix[Double], y: DenseMatrix[Double]): BufferedImage = {
    val canvas = new BufferedImage(D, D, BufferedImage.TYPE_INT_RGB)
    val wr = canvas.getRaster
    val mx = max(x)
    val my = max(y)
    //println("Max x is "+mx+" and max y is "+my)
    //println("Sum x is "+sum(x)+" and sum y is "+sum(y))
    for (i <- 0 until D) {
      for (j <- 0 until D) {
        wr.setSample(i, j, 2, round(255 * x(i, j) / mx).toInt) // band 2 is blue
        wr.setSample(i, j, 0, round(255 * y(i, j) / my).toInt) // band 0 is red
      }
    }
    canvas
  }

  def main(args: Array[String]): Unit = {
    println("Hello")
    println(S.toString)
    @tailrec
    def go(x: DenseMatrix[Double], y: DenseMatrix[Double], left: Int): Unit = {
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

