/*
examples.scala
 
MRF simulation code examples

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import breeze.linalg.{Vector => BVec, *}
import breeze.numerics.*
import breeze.stats.distributions.Rand.VariableSeed.randBasis
import breeze.stats.distributions.{Gaussian, Uniform}


// Ising model Gibbs sampler
object IsingGibbs extends IOApp.Simple:
  import Mrf.*
  def run: IO[Unit] =
    import breeze.stats.distributions.{Binomial,Bernoulli}
    val beta = 0.4
    val bdm = DenseMatrix.tabulate(500,600){
      case (i,j) => (new Binomial(1,0.2)).draw()
    }.map(_*2 - 1) // random matrix of +/-1s
    val pim0 = PImage(0,0,BDM2I(bdm))
    def gibbsKernel(pi: PImage[Int]): Int =
      val sum = pi.up.extract+pi.down.extract+pi.left.extract+pi.right.extract
      val p1 = math.exp(beta*sum)
      val p2 = math.exp(-beta*sum)
      val probplus = p1/(p1+p2)
      if (new Bernoulli(probplus).draw()) 1 else -1
    def oddKernel(pi: PImage[Int]): Int =
      if ((pi.x+pi.y) % 2 != 0) pi.extract else gibbsKernel(pi)
    def evenKernel(pi: PImage[Int]): Int =
      if ((pi.x+pi.y) % 2 == 0) pi.extract else gibbsKernel(pi)
    //def pims = LazyList.iterate(pim0)(_.coflatMap(gibbsKernel))
    def pims = LazyList.iterate(pim0)(_.coflatMap(oddKernel).coflatMap(evenKernel))
    plotFields(pims.take(50))

// Potts model Gibbs sampler
object PottsGibbs extends IOApp.Simple:
  import Mrf.*
  def run: IO[Unit] =
    import breeze.stats.distributions.Multinomial
    val beta = 0.2 // coupling constant
    val q = 4 // number of colours
    val p0 = DenseVector.fill(q)(1.0)
    val bdm = DenseMatrix.tabulate(500,600){
      case (i,j) => (new Multinomial(p0)).draw()}
    val pim0 = PImage(0,0,BDM2I(bdm))
    def matching(pi: PImage[Int])(i: Int): Int =
      (if (pi.up.extract == i) 1 else 0) +
      (if (pi.down.extract == i) 1 else 0) +
      (if (pi.left.extract == i) 1 else 0) +
      (if (pi.right.extract == i) 1 else 0)
    def gibbsKernel(pi: PImage[Int]): Int =
      val p = DenseVector.tabulate(q){
        case i => matching(pi)(i).toDouble }
      (new Multinomial(exp(p))).draw()
    def oddKernel(pi: PImage[Int]): Int =
      if ((pi.x+pi.y) % 2 != 0) pi.extract else gibbsKernel(pi)
    def evenKernel(pi: PImage[Int]): Int =
      if ((pi.x+pi.y) % 2 == 0) pi.extract else gibbsKernel(pi)
    //def pims = LazyList.iterate(pim0)(_.coflatMap(gibbsKernel))
    def pims = LazyList.iterate(pim0)(_.coflatMap(oddKernel).coflatMap(evenKernel))
    plotFields(pims.take(50))

// GMRF model Gibbs sampler
object GmrfGibbs extends IOApp.Simple:
  import Mrf.*
  def run: IO[Unit] =
    val beta = 0.25
    val bdm = DenseMatrix.tabulate(500,600){
      case (i,j) => Gaussian(0,1.0).draw()
    } // random init
    val pim0 = PImage(0,0,BDM2I(bdm))
    def gibbsKernel(pi: PImage[Double]): Double =
      val sum = pi.up.extract+pi.down.extract+pi.left.extract+pi.right.extract
      Gaussian(beta*sum, 1.0).draw()
    def oddKernel(pi: PImage[Double]): Double =
      if ((pi.x+pi.y) % 2 != 0) pi.extract else gibbsKernel(pi)
    def evenKernel(pi: PImage[Double]): Double =
      if ((pi.x+pi.y) % 2 == 0) pi.extract else gibbsKernel(pi)
    //def pims = LazyList.iterate(pim0)(_.coflatMap(gibbsKernel))
    def pims = LazyList.iterate(pim0)(_.coflatMap(oddKernel).coflatMap(evenKernel))
    plotFields(pims.take(100))

// Quartic MRF model sampler - MH version
object QuartMrfMh extends IOApp.Simple:
  import Mrf.*
  def run: IO[Unit] =
    val w = 0.45
    //val bdm = DenseMatrix.tabulate(1080, 1920){
    val bdm = DenseMatrix.tabulate(500, 700){
    //val bdm = DenseMatrix.tabulate(200, 300){
      case (i,j) => Gaussian(0, 1.0).draw()
    } // random init
    val pim0 = PImage(0, 0, BDM2I(bdm))
    def v(l: Double)(x: Double): Double = l*x - 2*x*x + x*x*x*x
    def mhKernel(pi: PImage[Double]): Double =
      val sum = pi.up.extract + pi.down.extract + pi.left.extract + pi.right.extract
      val x0 = pi.extract
      val x1 = x0 + Gaussian(0.0, 1.0).draw() // tune this!
      val lap = v(-w*sum)(x0) - v(-w*sum)(x1)
      if (math.log(Uniform(0, 1).draw()) < lap) x1 else x0
    def oddKernel(pi: PImage[Double]): Double =
      if ((pi.x+pi.y) % 2 != 0) pi.extract else mhKernel(pi)
    def evenKernel(pi: PImage[Double]): Double =
      if ((pi.x+pi.y) % 2 == 0) pi.extract else mhKernel(pi)
    //def pims = LazyList.iterate(pim0)(_.coflatMap(mhKernel))
    def pims = LazyList.iterate(pim0)(_.coflatMap(oddKernel).coflatMap(evenKernel))
    //plotFields(pims.thin(100).take(3000), showPlots=false, saveFrames=false)
    plotFields(pims.thin(50).take(50), showPlots=false, saveFrames=true)

// Quartic MRF model sampler - HMC version
object QuartMrfHmc extends IOApp.Simple:
  import Mrf.*
  def run: IO[Unit] =
    val w = 0.45
    //val bdm = DenseMatrix.tabulate(500, 600){
    val bdm = DenseMatrix.tabulate(200, 300){
      case (i,j) => Gaussian(0, 1.0).draw()
    } // random init
    val pim0 = PImage(0, 0, BDM2I(bdm))
    def v(x: Double): Double = -2*x*x + x*x*x*x
    def gv(x: Double): Double = -4*x + 4*x*x*x
    def lpi(pim: PImage[Double]): Double = pim.
      coflatMap(pim => w*pim.extract*(pim.right.extract + pim.down.extract) -
        v(pim.extract)).reduce(_+_)
    def glpi(pim: PImage[Double]): PImage[Double] = pim.
      coflatMap(pim => w*(pim.up.extract + pim.down.extract + pim.left.extract +
        pim.right.extract) - gv(pim.extract))
    val kern: PImage[Double] => PImage[Double] = hmcKernel(lpi, glpi, 0.01, 20)
    def pims = LazyList.iterate(pim0)(kern)
    plotFields(pims.thin(100).take(3000), showPlots=false, saveFrames=false)






// eof

