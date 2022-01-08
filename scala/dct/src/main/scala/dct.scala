/*
dct.scala

Messing around with trigonometric transforms in Scala

*/

import breeze.linalg.*
import breeze.numerics.*
import breeze.signal.*
import breeze.math.*
import math.Pi

object DCT:

  // Naive O(N^2) DFT implementation for test/reference
  def dftc(x: DenseVector[Complex], inverse: Boolean): DenseVector[Complex] =
    val N = x.length
    val ff = DenseVector.tabulate(N){n => (if (inverse) 1 else -1)*2*Pi*i*n/N}
    val X = DenseVector.tabulate(N){k => sum(x *:* exp(ff*Complex(k,0)))}
    if (inverse) X / Complex(N,0) else X

  def dft(x: DenseVector[Double], inverse: Boolean = false): DenseVector[Complex] =
    dftc(x map (Complex(_,0)), inverse)

  def idft(x: DenseVector[Complex]): DenseVector[Complex] = dftc(x, true)

  // iDFT from DFT, for fun...
  def idft0(x: DenseVector[Complex]): DenseVector[Complex] =
    val N = x.length
    val xr = x(x.length-1 to 0 by -1)
    val Xr = dftc(xr, false)
    val ff = DenseVector.tabulate(N){k => exp(-2*Pi*i*k/N)}
    (Xr *:* ff) / Complex(N,0)

  // Simple recursive FFT for sequences of length a power of 2
  // Not efficient - just testing understanding
  def fft(x: DenseVector[Complex]): DenseVector[Complex] =
    val N = x.length
    if (N == 1)
      x
    else
      if (N % 2 != 0) println("ERROR: Length not a power of 2. Result will be incorrect.")
      val ff = DenseVector.tabulate(N/2){k => exp(-2*Pi*i*k/N)}
      val E = fft(x(0 to N-1 by 2))
      val O = fft(x(1 to N-1 by 2))
      DenseVector.vertcat(E + (ff *:* O), E - (ff *:* O))

  // Naive DCT implementation for reference (O(n^2))
  def dct0(x: DenseVector[Double]): DenseVector[Double] =
    val N = x.length
    val cf = DenseVector.tabulate(N){n => Pi*(n + 0.5)/N}
    val X = DenseVector.tabulate(N){k => sum(x *:* cos(cf * k.toDouble))}
    X * (2.0 / N)

  // DCT implementation using the FFT from Breeze (O(n log n))
  def dct(x: DenseVector[Double]): DenseVector[Double] =
    val N = x.length
    val y = DenseVector.vertcat(x, x(N-1 to 0 by -1))
    val Y = fourierTr(y)
    val ff = DenseVector.tabulate(2*N){k => -Pi*i*k/(2*N)}
    val sY = Y *:* exp(ff)
    sY(0 to N-1).map(_.real) / N.toDouble

  // Naive iDCT implementation for reference (O(n^2))
  def idct0(x: DenseVector[Double]): DenseVector[Double] =
    val N = x.length
    val cf = DenseVector.tabulate(N-1){k => Pi*(k+1)/N}
    DenseVector.tabulate(N){n => x(0)/2 + sum(x(1 to N-1) *:* cos(cf * (n+0.5)))}

  // iDCT implementation using the FFT from Breeze (O(n log n))
  def idct(x: DenseVector[Double]): DenseVector[Double] =
    val N = x.length
    val y = DenseVector.vertcat(x, DenseVector(0.0), -x(N-1 to 1 by -1))
    val ff = DenseVector.tabulate(2*N){k => Pi*i*k/(2*N)}
    val sy = y.map(Complex(_,0.0)) *:* exp(ff)
    val Y = iFourierTr(sy)
    Y(0 to N-1).map(_.real) * N.toDouble

  // 2d DCT and inversion using 1d on rows and columns
  // TODO: this is trivial to parallelise
  def dct2(X: DenseMatrix[Double], inverse: Boolean = false): DenseMatrix[Double] =
    val x = X.copy
    (0 until x.rows).foreach{j =>
      x(j, ::) := (if (inverse) idct(x(j, ::).t)
        else dct(x(j, ::).t)).t}
    (0 until x.cols).foreach{k =>
      x(::, k) := (if (inverse) idct(x(::, k))
        else dct(x(::, k)))}
    x

object Examp:

  import DCT.*

  @main def examples() =
    val x = DenseVector(1.0,2.0,3.0,2.0,4.0,3.0)
    println(x)
    // try breeze built-in fft
    val X = fourierTr(x)
    println("Breeze FFT")
    println(X)
    println("My dft")
    println(dft(x))
    println("My iDFT")
    println(idft(x map (Complex(_,0))))
    println("iDFT from DFT")
    println(idft0(x map (Complex(_,0))))
    println("Breeze iDFT")
    println(iFourierTr(x))
    println("Breeze inversion")
    println(iFourierTr(fourierTr(x)).map(_.real))
    println("My inversion")
    println(idft(dft(x)).map(_.real))
    println("My fft")
    val xc = DenseVector(1,2,3,2,4,3,4,5) map (Complex(_,0))
    println(fft(xc))
    println("Breeze FFT")
    println(fourierTr(xc))
    println("DCT0")
    println(dct0(x))
    println("DCT")
    println(dct(x))
    println("iDCT0")
    println(idct0(x))
    println("iDCT")
    println(idct(x))
    println("DCT0 Inversion")
    println(idct0(dct0(x)))
    println("DCT Inversion")
    println(idct(dct(x)))
    val M = DenseMatrix((1.0,2.0,3.0),(2.0,2.0,2.0))
    println(M)
    println(fourierTr(M))
    println(dct2(M))
    println(dct2(dct2(M), true))

object FBmExample:

  import DCT.*
  import breeze.stats.distributions.*
  import breeze.stats.distributions.Rand.VariableSeed.randBasis
  import java.awt.image.BufferedImage

  def dm2bi(m: DenseMatrix[Double]): BufferedImage =
    val canvas = new BufferedImage(m.cols, m.rows, BufferedImage.TYPE_INT_RGB)
    val wr = canvas.getRaster
    val mx = max(m)
    val mn = min(m)
    (0 until m.cols).foreach { x =>
      (0 until m.rows).foreach { y =>
        val shade = round(255 * (m(y, x) - mn) / (mx - mn)).toInt
        wr.setSample(x, y, 0, shade) // R
        wr.setSample(x, y, 1, shade) // G
        wr.setSample(x, y, 2, 255) // B
      }
    }
    canvas

  def showImage(m: DenseMatrix[Double]): Unit =
    import breeze.plot.*
    val fig = Figure("fBm")
    fig.width = 1200
    fig.height = 1200
    val p0 = fig.subplot(0)
    p0 += image(m)
    fig.refresh()

  @main def fbmex() =
    val N = 1024
    val H = 0.9
    val sd = DenseMatrix.tabulate(N, N){(j, k) =>
      if (j*j + k*k < 9) 0.0 else
        math.pow(j*j + k*k, -(H + 1)/2) }
    val M = sd map (s => Gaussian(0.0, s).draw())
    val m = dct2(M, true)
    javax.imageio.ImageIO.write(dm2bi(m), "png", new java.io.File("fBm.png"))
    showImage(m)


// eof

