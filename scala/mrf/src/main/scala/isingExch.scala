/*
isingExch.scala
 
Exchange algorithm for MCMC inference for the temperature of an Ising model

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import breeze.linalg.{Vector => BVec, *}
import breeze.numerics.*
import breeze.stats.distributions.Rand.VariableSeed.randBasis
import breeze.stats.distributions.{Gaussian, Uniform}

object IsingExch extends IOApp.Simple:
  import Mrf.*

  def readImage(fileName: String): IO[Image[Int]] = IO {
    val im = javax.imageio.ImageIO.read(new java.io.File("ising03.png"))
    val h = im.getHeight()
    val w = im.getWidth()
    val ras = im.getData()
    val bdm = DenseMatrix.tabulate(h, w){ case (i, j) => ras.getSample(j, i, 0) }
    val imi = BDM2I(bdm.map(p => if (p == 0) -1 else 1)) // specific to Ising model
    //javax.imageio.ImageIO.write(Image.mkImage(imi), "png", new java.io.File("test"+".png"))
    imi
    }





  def run: IO[Unit] =
    val init = 0.1 // initial temperature guess
    val its = 100 // number of MCMC iterations
    val th = 2 // thinning interval of MCMC chain
    val num = 10 // number of MCMC iters to generate a "perfect" sample...
    for
      _ <- IO.println("Exchange algorithm for an Ising model")
      im <- readImage("ising03.png")
      pim = PImage(0, 0, im)

      _ <- IO.println("Goodbye.")
    yield ()




// eof

