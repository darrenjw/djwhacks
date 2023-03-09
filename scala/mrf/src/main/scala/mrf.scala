/*
mrf.scala
Stub for Scala Cats code
*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.distributions.Rand.VariableSeed.randBasis
import breeze.stats.distributions.{Gaussian, Uniform}

import scala.collection.parallel.immutable.ParVector
import scala.collection.parallel.CollectionConverters.*

import annotation.tailrec


object Mrf:

  // Basic image class
  case class Image[T](w: Int, h: Int, data: ParVector[T]) {
  //case class Image[T](w: Int, h: Int, data: Vector[T]) {
    def apply(x: Int, y: Int): T = data(x*h+y)
    def map[S](f: T => S): Image[S] = Image(w, h, data map f)
    def updated(x: Int, y: Int, value: T): Image[T] =
      Image(w, h, data.updated(x*h+y, value))
  }

  // Pointed image (with a focus/cursor)
  case class PImage[T](x: Int, y: Int, image: Image[T]) {
    def extract: T = image(x, y)
    def map[S](f: T => S): PImage[S] = PImage(x, y, image map f)
    def coflatMap[S](f: PImage[T] => S): PImage[S] = PImage(
      x, y, Image(image.w, image.h,
      (0 until (image.w * image.h)).toVector.par.map(i => {
      //(0 until (image.w * image.h)).toVector.map(i => {
        val xx = i / image.h
        val yy = i % image.h
        f(PImage(xx, yy, image))
      })))
    // now a few methods for navigation - not part of the comonad interface
    // using periodic boundary conditions
    def up: PImage[T] = {
      val py = y-1
      val ny = if (py >= 0) py else (py + image.h)
      PImage(x, ny, image)
    }
    def down: PImage[T] = {
      val py = y+1
      val ny = if (py < image.h) py else (py - image.h)
      PImage(x, ny, image)
    }
    def left: PImage[T] = {
      val px = x-1
      val nx = if (px >= 0) px else (px + image.w)
      PImage(nx, y, image)
    }
    def right: PImage[T] = {
      val px = x+1
      val nx = if (px < image.w) px else (px - image.w)
      PImage(nx, y, image)
    }
  }

  // Provide evidence that PImage is a Cats Comonad
  given Comonad[PImage] with
    def extract[A](wa: PImage[A]) = wa.extract
    def coflatMap[A,B](wa: PImage[A])(f: PImage[A] => B): PImage[B] =
      wa.coflatMap(f)
    def map[A,B](wa: PImage[A])(f: A => B): PImage[B] = wa.map(f)

  // Provide evidence that PImage is a Cats Apply
  given Apply[PImage] with
    def map[A,B](wa: PImage[A])(f: A => B): PImage[B] = wa.map(f)
    def ap[A, B](ff: PImage[A=>B])(fa: PImage[A]): PImage[B] =
      PImage(ff.x, ff.y, Image(ff.image.w, ff.image.h, (ff.image.data zip fa.image.data).map((ffi, fai) => ffi(fai))))

  // Provide evidence that PImage is a Cats Reducible
  given Reducible[PImage] with
    def foldLeft[A, B](fa: PImage[A], b: B)(f: (B, A) => B): B =
      fa.image.data.foldLeft(b)(f)
    def foldRight[A, B](fa: PImage[A], lb: Eval[B])(f: (A, Eval[B]) => Eval[B]): Eval[B] =
      fa.image.data.foldRight(lb)(f)
    def reduceLeftTo[A, B](fa: PImage[A])(f: A => B)(g: (B, A) => B): B =
      fa.image.data.tail.foldLeft(f(fa.image.data.head))(g)
    def reduceRightTo[A, B](fa: PImage[A])(f: A => B)(g: (A, Eval[B]) => Eval[B]): Eval[B] =
      fa.image.data.init.foldRight(Eval.later(f(fa.image.data.last)))(g)


  // convert to and from Breeze matrices
  import breeze.linalg.{Vector => BVec, _}
  def BDM2I[T](m: DenseMatrix[T]): Image[T] =
    Image(m.cols, m.rows, m.data.toVector.par)
    //Image(m.cols, m.rows, m.data.toVector)
  def I2BDM(im: Image[Double]): DenseMatrix[Double] = 
    new DenseMatrix(im.h,im.w,im.data.toArray)


  def mhKern[S](
      logPost: S => Double, rprop: S => S,
      dprop: (S, S) => Double = (n: S, o: S) => 1.0
    ): (S) => S =
      val r = Uniform(0.0,1.0)
      x0 =>
        val x = rprop(x0)
        val ll0 = logPost(x0)
        val ll = logPost(x)
        val a = ll - ll0 + dprop(x0, x) - dprop(x, x0)
        if (math.log(r.draw()) < a) x else x0

  def hmcKernel[F[_]: Apply](lpi: F[Double] => Double, glpi: F[Double] => F[Double],
      eps: Double = 1e-4, l: Int = 10)(using Reducible[F]): F[Double] => F[Double] =
    def add(p: F[Double], q: F[Double]): F[Double] = (p product q) map ((pi, qi) => pi + qi)
    def scale(s: Double, p: F[Double]): F[Double] = p map (pi => s * pi)
    def leapf(q: F[Double], p: F[Double]): (F[Double], F[Double]) =
      @tailrec def go(q0: F[Double], p0: F[Double], l: Int): (F[Double], F[Double]) =
        val q = add(q0, scale(eps, p0))
        val p = if (l > 1)
          add(p0, scale(eps, glpi(q)))
        else
          add(p0, scale(0.5*eps, glpi(q)))
        if (l == 1)
          (q, p)
        else
          go(q, p, l-1)
      go(q, add(p, scale(0.5*eps, glpi(q))), l)
    def alpi(x: (F[Double], F[Double])): Double =
      val (q, p) = x
      lpi(q) - 0.5*(p.map(pi => pi*pi).reduce(_+_))
    def rprop(x: (F[Double], F[Double])): (F[Double], F[Double]) =
      val (q, p) = x
      leapf(q, p)
    val mhk = mhKern(alpi, rprop)
    (q: F[Double]) =>
      val p = q map (qi => Gaussian(0, 1.0).draw())
      mhk((q, p))._1




// Examples...



object IsingGibbs extends IOApp.Simple:

  import Mrf.*
 
  // Ising model Gibbs sampler
  def run: IO[Unit] = IO {
    import breeze.stats.distributions.{Binomial,Bernoulli}
    val beta = 0.4
    val bdm = DenseMatrix.tabulate(500,600){
      case (i,j) => (new Binomial(1,0.2)).draw()
    }.map(_*2 - 1) // random matrix of +/-1s
    val pim0 = PImage(0,0,BDM2I(bdm))
    def gibbsKernel(pi: PImage[Int]): Int = {
      val sum = pi.up.extract+pi.down.extract+pi.left.extract+pi.right.extract
      val p1 = math.exp(beta*sum)
      val p2 = math.exp(-beta*sum)
      val probplus = p1/(p1+p2)
      if (new Bernoulli(probplus).draw()) 1 else -1
    }
    def oddKernel(pi: PImage[Int]): Int =
      if ((pi.x+pi.y) % 2 != 0) pi.extract else gibbsKernel(pi)
    def evenKernel(pi: PImage[Int]): Int =
      if ((pi.x+pi.y) % 2 == 0) pi.extract else gibbsKernel(pi)
    //def pims = LazyList.iterate(pim0)(_.coflatMap(gibbsKernel))
    def pims = LazyList.iterate(pim0)(_.coflatMap(oddKernel).coflatMap(evenKernel))
    // render
    import breeze.plot.*
    val fig = Figure("MRF sampler")
    //fig.visible = false
    fig.width = 1000
    fig.height = 800
    pims.take(50).zipWithIndex.foreach{case (pim,i) => {
      print(s"$i ")
      fig.clear()
      val p = fig.subplot(1,1,0)
      p.title = s"MRF: frame $i"
      p += image(I2BDM(pim.image.map{_.toDouble}))
      fig.refresh()
      //fig.saveas(f"mrf$i%04d.png")
    }}
    println()
  }

object GmrfGibbs extends IOApp.Simple:

  import Mrf.*
 
  // GMRF model Gibbs sampler
  def run: IO[Unit] = IO {
    import breeze.stats.distributions.{Gaussian}
    import breeze.stats.*
    val beta = 0.25
    val bdm = DenseMatrix.tabulate(500,600){
      case (i,j) => Gaussian(0,1.0).draw()
    } // random init
    val pim0 = PImage(0,0,BDM2I(bdm))
    def gibbsKernel(pi: PImage[Double]): Double = {
      val sum = pi.up.extract+pi.down.extract+pi.left.extract+pi.right.extract
      Gaussian(beta*sum, 1.0).draw()
    }
    def oddKernel(pi: PImage[Double]): Double =
      if ((pi.x+pi.y) % 2 != 0) pi.extract else gibbsKernel(pi)
    def evenKernel(pi: PImage[Double]): Double =
      if ((pi.x+pi.y) % 2 == 0) pi.extract else gibbsKernel(pi)
    //def pims = LazyList.iterate(pim0)(_.coflatMap(gibbsKernel))
    def pims = LazyList.iterate(pim0)(_.coflatMap(oddKernel).coflatMap(evenKernel))
    // render
    import breeze.plot.*
    val fig = Figure("MRF sampler")
    //fig.visible = false
    fig.width = 1000
    fig.height = 800
    pims.take(200).zipWithIndex.foreach{case (pim,i) => {
      print(s"$i ")
      fig.clear()
      val p = fig.subplot(1,1,0)
      p.title = s"MRF: frame $i"
      val mat = I2BDM(pim.image)
      p += image(mat)
      fig.refresh()
      println(" "+mean(mat)+" "+(max(mat) - min(mat)))
      //fig.saveas(f"mrf$i%04d.png")
    }}
    //println()
  }


object QuartMrfMh extends IOApp.Simple:

  import Mrf.*
 
  // Quartic MRF model sampler
  def run: IO[Unit] = IO {
    import breeze.stats.*
    val w = 0.5
    val bdm = DenseMatrix.tabulate(500,600){
      case (i,j) => Gaussian(0,1.0).draw()
    } // random init
    val pim0 = PImage(0,0,BDM2I(bdm))
    def v(l: Double)(x: Double): Double = l*x - 2*x*x + x*x*x*x
    def mhKernel(pi: PImage[Double]): Double = {
      val sum = pi.up.extract+pi.down.extract+pi.left.extract+pi.right.extract
      val x0 = pi.extract
      val x1 = x0 + Gaussian(0.0, 1.0).draw() // tune this!
      val lap = v(-w*sum)(x0) - v(-w*sum)(x1)
      if (math.log(Uniform(0,1).draw()) < lap) x1 else x0
    }
    def oddKernel(pi: PImage[Double]): Double =
      if ((pi.x+pi.y) % 2 != 0) pi.extract else mhKernel(pi)
    def evenKernel(pi: PImage[Double]): Double =
      if ((pi.x+pi.y) % 2 == 0) pi.extract else mhKernel(pi)
    //def pims = LazyList.iterate(pim0)(_.coflatMap(mhKernel))
    def pims = LazyList.iterate(pim0)(_.coflatMap(oddKernel).coflatMap(evenKernel))
    // render
    import breeze.plot.*
    val fig = Figure("MRF sampler")
    //fig.visible = false
    fig.width = 1000
    fig.height = 800
    pims.take(100).zipWithIndex.foreach{case (pim,i) => {
      print(s"$i ")
      fig.clear()
      val p = fig.subplot(1,1,0)
      p.title = s"MRF: frame $i"
      val mat = I2BDM(pim.image)
      p += image(mat)
      fig.refresh()
      println(" "+mean(mat)+" "+(max(mat) - min(mat)))
      //fig.saveas(f"mrf$i%04d.png")
    }}
    //println()
  }


object QuartMrfHmc extends IOApp.Simple:

  import Mrf.*
 
  // Quartic MRF model sampler - HMC version
  def run: IO[Unit] = IO {
    import breeze.stats.*
    val w = 0.5
    val bdm = DenseMatrix.tabulate(500,600){
      case (i,j) => Gaussian(0,1.0).draw()
    } // random init
    val pim0 = PImage(0,0,BDM2I(bdm))
    def v(x: Double): Double = -2*x*x + x*x*x*x
    def gv(x: Double): Double = -4*x + 4*x*x*x
    def lpi(pim: PImage[Double]): Double = pim.
      coflatMap(pim => w*pim.extract*(pim.right.extract + pim.down.extract) - v(pim.extract)).
        reduce(_+_)
    def glpi(pim: PImage[Double]): PImage[Double] = pim.
      coflatMap(pim => w*(pim.up.extract+pim.down.extract+pim.left.extract+pim.right.extract) -
        gv(pim.extract))
    val kern: PImage[Double] => PImage[Double] = hmcKernel(lpi, glpi, 0.01, 100)
    def pims = LazyList.iterate(pim0)(kern)
    // render
    import breeze.plot.*
    val fig = Figure("MRF sampler")
    //fig.visible = false
    fig.width = 1000
    fig.height = 800
    pims.take(500).zipWithIndex.foreach{case (pim,i) => {
      print(s"$i ")
      fig.clear()
      val p = fig.subplot(1,1,0)
      p.title = s"MRF: frame $i"
      val mat = I2BDM(pim.image)
      p += image(mat)
      fig.refresh()
      println(" "+mean(mat)+" "+(max(mat) - min(mat)))
      //fig.saveas(f"mrf$i%04d.png")
    }}
    //println()
  }


