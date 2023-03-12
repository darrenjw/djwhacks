/*
mrf.scala
 
MRF simulation code

*/

import cats.*
import cats.implicits.*
import cats.effect.IO

import breeze.linalg.{Vector => BVec, *}
import breeze.numerics.*
import breeze.stats.distributions.Rand.VariableSeed.randBasis
import breeze.stats.distributions.{Gaussian, Uniform}

import scala.collection.parallel.immutable.ParVector
import scala.collection.parallel.CollectionConverters.*

import java.awt.image.BufferedImage

import annotation.tailrec

object Mrf:

  // Basic image class
  case class Image[T](w: Int, h: Int, data: ParVector[T]):
  //case class Image[T](w: Int, h: Int, data: Vector[T]) {
    def apply(x: Int, y: Int): T = data(x*h+y)
    def map[S](f: T => S): Image[S] = Image(w, h, data map f)
    def updated(x: Int, y: Int, value: T): Image[T] =
      Image(w, h, data.updated(x*h+y, value))

  case object Image:
    def mkImage[A: Numeric](im: Image[A]): BufferedImage =
      import math.Numeric.Implicits.infixNumericOps
      val canvas = new BufferedImage(im.w, im.h, BufferedImage.TYPE_INT_RGB)
      val wr = canvas.getRaster
      val mx = im.data.map(_.toDouble).reduce((x, y) => math.max(x,y))
      val mn = im.data.map(_.toDouble).reduce((x, y) => math.min(x,y))
      for (i <- 0 until im.w)
        for (j <- 0 until im.h)
          val level = round(255 * (im(i, j).toDouble - mn) / (mx - mn)).toInt
          wr.setSample(i, j, 0, level)
          wr.setSample(i, j, 1, level)
          wr.setSample(i, j, 2, 255)
      canvas
    def saveImage[A: Numeric](im: Image[A], fileName: String): Unit =
      javax.imageio.ImageIO.write(mkImage(im), "png", new java.io.File(fileName+".png"))

  // Pointed image (with a focus/cursor) - comonadic
  case class PImage[T](x: Int, y: Int, image: Image[T]):
    def extract: T = image(x, y)
    def map[S](f: T => S): PImage[S] = PImage(x, y, image map f)
    def coflatMap[S](f: PImage[T] => S): PImage[S] = PImage(
      x, y, Image(image.w, image.h,
      (0 until (image.w * image.h)).toVector.par.map(i =>
      //(0 until (image.w * image.h)).toVector.map(i =>
        val xx = i / image.h
        val yy = i % image.h
        f(PImage(xx, yy, image))
      )))
    // now a few methods for navigation - not part of the comonad interface
    // using periodic boundary conditions
    def up: PImage[T] =
      val py = y-1
      val ny = if (py >= 0) py else (py + image.h)
      PImage(x, ny, image)
    def down: PImage[T] =
      val py = y+1
      val ny = if (py < image.h) py else (py - image.h)
      PImage(x, ny, image)
    def left: PImage[T] =
      val px = x-1
      val nx = if (px >= 0) px else (px + image.w)
      PImage(nx, y, image)
    def right: PImage[T] =
      val px = x+1
      val nx = if (px < image.w) px else (px - image.w)
      PImage(nx, y, image)

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
      PImage(ff.x, ff.y,
        Image(ff.image.w, ff.image.h,
          (ff.image.data zip fa.image.data).map((ffi, fai) => ffi(fai))))

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
    // optionally override reduce
    def reduce[A](fa: PImage[A])(f: (A, A) => A): A = fa.image.data.reduce(f)

  // convert to and from Breeze matrices
  def BDM2I[T](m: DenseMatrix[T]): Image[T] =
    Image(m.cols, m.rows, m.data.toVector.par)
    //Image(m.cols, m.rows, m.data.toVector)
  def I2BDM(im: Image[Double]): DenseMatrix[Double] = 
    new DenseMatrix(im.h,im.w,im.data.toArray)

  // Thinnable typeclass and instance for LazyLists
  trait Thinnable[F[_]]:
    extension [T](ft: F[T])
      def thin(th: Int): F[T]

  given Thinnable[LazyList] with
    extension [T](s: LazyList[T])
      def thin(th: Int): LazyList[T] =
        val ss = s.drop(th-1)
        if (ss.isEmpty) LazyList.empty else
          ss.head #:: ss.tail.thin(th)

  // a MH kernel as needed for HMC
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

  // a HMC kernel
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


  // Impure code below (wrapped in IO)

  def plotFields[A: Numeric](s: LazyList[PImage[A]], showPlots: Boolean = true,
      savePixels: Boolean = true, saveFrames: Boolean = false): IO[Unit] = IO {
    import breeze.plot.*
    import breeze.stats.*
    import math.Numeric.Implicits.infixNumericOps
    val fig = Figure("MRF sampler")
    if (showPlots)
      fig.width = 1000
      fig.height = 800
    else
      fig.visible = false
    val fs = new java.io.FileWriter("mrf.csv")
    fs.write("P1,P2,P3\n")
    s.zipWithIndex.foreach{case (pim,i) =>
      print(s"$i ")
      val mat = I2BDM(pim.image.map(_.toDouble))
      println(" mean:"+mean(mat)+" range:"+(max(mat) - min(mat)))
      if (showPlots)
        fig.clear()
        val p = fig.subplot(1,1,0)
        p.title = s"MRF: frame $i"
        p += image(mat)
        fig.refresh()
      if (saveFrames)
        Image.saveImage(pim.image, f"mrf-$i%04d")
      if (savePixels)
        fs.write(""+mat(0,0)+","+mat(0,1)+","+mat(10,10)+"\n")
    }
    fs.close()
  }



// eof

