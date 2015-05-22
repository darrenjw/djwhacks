/*
mandel.scala

*/

import breeze.math.Complex
import java.awt.image.BufferedImage
import java.awt.{Graphics2D,Color,Font,BasicStroke}
import java.awt.geom._


object Main {

def mandel(c: Complex,maxIt:Int=255): Int = {
  def go(z: Complex, its: Int): Int = {
    if (its==maxIt) 0 else {
      if (z.abs>2.0) its else go(z*z+c,its+1)
    }
  }
  go(Complex(0,0),0)
}

def coords(pix: Int): IndexedSeq[(Int,Int)] = {
  for {
    i <- 0 until pix
    j <- 0 until pix
  } yield (i,j)
}

def coord2c(coord: (Int,Int),centre: Complex, size: Double, pix: Int): Complex = coord match {
  case (i,j) => Complex(i-pix/2,j-pix/2)*(size/pix)+centre
}

def main(args: Array[String]): Unit = {
  val centre=Complex(-0.5,0.0)
  val size=3.0
  val pix=1500
  val co=coords(pix)
  val cvals=co map {coord2c(_,centre,size,pix)}
  val m=cvals map {mandel(_)} // could combine these maps
  val ms=co zip m
  val canvas = new BufferedImage(pix, pix, BufferedImage.TYPE_BYTE_GRAY)
  val wr=canvas.getRaster()
  wr.setSample(10,10,0,128)
  for (m <- ms) {
    wr.setSample(m._1._1,m._1._2,0,m._2)
  }
  javax.imageio.ImageIO.write(canvas, "png", new java.io.File("mandel.png"))
}


}


// eof


