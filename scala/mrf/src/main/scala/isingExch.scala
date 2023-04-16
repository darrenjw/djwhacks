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
import breeze.stats.distributions.*

object IsingExch extends IOApp.Simple:
  import Mrf.{given, *}

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

  def gibbsKernel(beta: Double)(pi: PImage[Int]): Int =
    val sum = pi.up.extract+pi.down.extract+pi.left.extract+pi.right.extract
    val p1 = math.exp(beta*sum)
    val p2 = math.exp(-beta*sum)
    val probplus = p1/(p1+p2)
    if (new Bernoulli(probplus).draw()) 1 else -1

  def oddKernel(beta: Double)(pi: PImage[Int]): Int =
    if ((pi.x+pi.y) % 2 != 0) pi.extract else gibbsKernel(beta)(pi)

  def evenKernel(beta: Double)(pi: PImage[Int]): Int =
    if ((pi.x+pi.y) % 2 == 0) pi.extract else gibbsKernel(beta)(pi)

  def update(beta: Double)(pi: PImage[Int]): PImage[Int] =
    pi.coflatMap(oddKernel(beta)).coflatMap(evenKernel(beta))

  @annotation.tailrec
  def perfect(beta: Double, pi: PImage[Int], its: Int = 10): PImage[Int] =
    if (its == 0) pi else perfect(beta, update(beta)(pi), its-1)      

  def mhUpdate(data: PImage[Int])(state: (Double, PImage[Int])): (Double, PImage[Int]) =
    val (bo, po) = state
    //println(bo)
    val bn = Gaussian(bo, 0.01).draw()
    val pn = perfect(bn, po)
    def lpi(beta: Double, pim: PImage[Int]): Double = pim.
      coflatMap(pim => beta*pim.extract*(pim.right.extract + pim.down.extract)).
        reduce(_+_)
    val a = lpi(bn, data) + lpi(bo, pn) - lpi(bo, data) - lpi(bn, pn)
    if (log(Uniform(0,1).draw()) < a)
      (bn, pn)
    else
      (bo, po)

  def processChain(ch: LazyList[Double]): IO[Unit] =
    val chain = ch.drop(100).thin(5).take(10000) // burn, thin and its specified here
    IO {
      val fs = new java.io.FileWriter("exch.csv")
      fs.write("beta\n")
      chain.foreach(it =>
        fs.write(s"$it\n")
      )
      fs.close()
    } 

  def run: IO[Unit] =
    val init = 0.1 // initial temperature guess
    for
      _ <- IO.println("Exchange algorithm for an Ising model")
      im <- readImage("ising03.png")
      data = PImage(0, 0, im)
      kern = mhUpdate(data)
      mcmc = LazyList.iterate((init, data))(kern)
      _ <- IO.println("Running MCMC now...")
      _ <- processChain(mcmc.map(_._1))
      _ <- IO.println("MCMC finished. Goodbye.")
    yield ()




// eof

