/*
reactdiffsg.scala
2d reaction diffusion of the Lotka Volterra predator prey system
Spatial CLE approx

 */

object ReactDiff2dSG {

  import annotation.tailrec
  // import Math.sqrt

  import breeze.linalg._
  import breeze.math._
  import breeze.numerics._
  import breeze.stats.distributions.Gaussian

  import java.awt.image.BufferedImage

  val D = 50
  val T = 12 // Was 120...
  val dt = 0.25
  val th = Vector(1.0, 0.005, 0.6)
  val dc = 0.25

  val N = (T / dt).toInt
  val x = DenseMatrix.zeros[Double](D, D)
  x(D / 2, D / 2) = 60
  val y = DenseMatrix.zeros[Double](D, D)
  y(D / 2, D / 2) = 60

  def up(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    DenseMatrix.vertcat(m(1 until m.rows, ::), m(0 to 0, ::))
  }

  def down(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    DenseMatrix.vertcat(m((m.rows - 1) to (m.rows - 1), ::), m(0 until (m.rows - 1),::))
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
    // abs(m) // reflect
    m map { e => if (e>0.0) e else 0.0 } // absorb
}

  def sqrt(m: DenseMatrix[Double]): DenseMatrix[Double] = m map (Math.sqrt(_))

  def diffuse(m: DenseMatrix[Double]): DenseMatrix[Double] = {
    val dwt = new DenseMatrix(D,D,Gaussian(0.0,Math.sqrt(dt)).sample(D*D).toArray)
    val dwts = new DenseMatrix(D,D,Gaussian(0.0,Math.sqrt(dt)).sample(D*D).toArray)
    rectify(m + dc*laplace(m)*dt + Math.sqrt(dc)*(
      sqrt(m+left(m))*dwt - sqrt(m+right(m))*right(dwt)
        + sqrt(m+up(m))*dwts - sqrt(m+down(m))*down(dwts)
      )
    )
}

  def stepLV(x: DenseMatrix[Double],y: DenseMatrix[Double],dt: Double): (DenseMatrix[Double],DenseMatrix[Double]) = {
    (diffuse(x),diffuse(y))
}


  def mkImage(x: DenseMatrix[Double], y: DenseMatrix[Double]): BufferedImage = {
    val canvas = new BufferedImage(D, D, BufferedImage.TYPE_INT_RGB)
    val wr = canvas.getRaster
    val mx = max(x)
    val my = max(y)
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

