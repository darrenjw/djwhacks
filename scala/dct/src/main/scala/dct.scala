/*
Stub.scala
Stub for Scala Breeze code
*/

import breeze.linalg.*
import breeze.numerics.*
import breeze.math.*
import math.Pi

object RefImps:

  // Naive O(N^2) DFT implementation for test/reference
  def dftc(x: DenseVector[Complex], inverse: Boolean): DenseVector[Complex] =
    val N = x.length
    val ff = DenseVector.tabulate(N){n => (if (inverse) 1 else -1)*2*Pi*i*n/N}
    val X = DenseVector.tabulate(N){k => sum(x *:* exp(ff*Complex(k,0)))}
    if (inverse) X / Complex(N,0) else X

  def dft(x: DenseVector[Double], inverse: Boolean = false): DenseVector[Complex] =
    dftc(x.map(Complex(_,0)), inverse)

  def idft(x: DenseVector[Complex]): DenseVector[Complex] = dftc(x, true)

  def dct(x: DenseVector[Double]): DenseVector[Double] =
    val N = x.length
    val y = DenseVector.vertcat(x, x(N-1 to 0 by -1))
    val Y = dft(y) // TODO: Switch to an FFT at some point...
    val ff = DenseVector.tabulate(2*N){k => -Pi*i*k/(2*N)}
    val sY = Y *:* exp(ff)
    sY(0 to N-1).map(_.real) / N.toDouble

  def idct(x: DenseVector[Double]): DenseVector[Double] =
    val N = x.length
    val y = DenseVector.vertcat(x, DenseVector(0.0), -x(N-1 to 1 by -1))
    val ff = DenseVector.tabulate(2*N){k => Pi*i*k/(2*N)}
    val sy = y.map(Complex(_,0.0)) *:* exp(ff)
    val Y = idft(sy) // TODO: Switch to an FFT at some point
    Y(0 to N-1).map(_.real) * N.toDouble

  def idct2d(X: DenseMatrix[Double]): DenseMatrix[Double] = ???


  def main(args: Array[String]): Unit =
    val x = DenseVector(1.0,2.0,3.0,2.0,4.0,3.0)
    println(x)
    // try breeze built-in fft
    import breeze.signal.*
    val X = fourierTr(x)
    println("Breeze FFT")
    println(X)
    println("My dft")
    println(dft(x))
    println(iFourierTr(x))
    println("Breeze inversion")
    println(iFourierTr(fourierTr(x)).map(_.real))
    println("My inversion")
    println(idft(dft(x)).map(_.real))
    println("DCT")
    println(dct(x))
    println("Inversion")
    println(idct(dct(x)))
    // 2d stuff
    // try breeze
    val M = DenseMatrix((1.0,2.0,3.0),(2.0,2.0,2.0))
    println(M)
    println(fourierTr(M))


