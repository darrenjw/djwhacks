/*
swing.scala
Swing app to visualise langtons ant

 */

import scala.swing._
//import scala.swing.event._
import java.awt.{Graphics2D, Color, BasicStroke}
import java.awt.image.BufferedImage
//import scala.math._
import scala.util.Random
import LangtonsAnt._

object AntSwingApp extends SimpleSwingApplication {

  val ssize=250
  val stretch=5

  def top = new MainFrame {
    title = "Langton's Ant"
    val panel = ImagePanel(ssize*stretch)
    contents = new BoxPanel(Orientation.Vertical) {
      contents += panel
      border = Swing.EmptyBorder(10, 10, 10, 10)
    }
    var is=thinStream(stateStream(State(ssize)),100).map(s=>img2Image(s.img)).map(biResize(_,ssize*stretch,ssize*stretch))
    val timer=new javax.swing.Timer(1,Swing.ActionListener(e=>{
      panel.bi=is.head
      is=is.tail
      panel.repaint()
    }))
    timer.start()
  }

}

case class ImagePanel(var bi: BufferedImage) extends Panel {
  override def paintComponent(g: Graphics2D) = {
    //g.clearRect(0,0,size.width,size.height)
    g.drawImage(bi,0,0,null)
    }
}
object ImagePanel {
  def apply(s: Int) = {
    val bi=new BufferedImage(s,s,BufferedImage.TYPE_BYTE_BINARY)
    val ip=new ImagePanel(bi)
    ip.preferredSize = new Dimension(s,s)
    ip
    }
  }


/* eof */

