/*
swing.scala
Visualise an image stream with a Swing frame

 */

import scala.swing._
import java.awt.{Graphics2D, Color, BasicStroke}
import java.awt.image.BufferedImage
import scala.util.Random
import scala.swing.event.{ButtonClicked, WindowClosing}

class SwingImageViewer(var is: Stream[BufferedImage], timerDelay: Int) {

  def top = new MainFrame {
    title = "Swing Image Viewer"
    val start = new Button { text = "Start" }
    val stop = new Button { text = "Stop" }
    val panel = ImagePanel(is.head.getWidth, is.head.getHeight)
    contents = new BoxPanel(Orientation.Vertical) {
      contents += start
      contents += stop
      contents += panel
      border = Swing.EmptyBorder(10, 10, 10, 10)
    }
    peer.setDefaultCloseOperation(0)
    listenTo(start)
    listenTo(stop)
    val timer = new javax.swing.Timer(timerDelay, Swing.ActionListener(e => {
      panel.bi = is.head
      is = is.tail
      panel.repaint()
    }))
    reactions += {
      case ButtonClicked(b) => {
        if (b.text == "Start")
          timer.start()
        else
          timer.stop()
      }
      case WindowClosing(_) => {
        println("Close button clicked. Exiting...")
        sys.exit()
      }
    }
  }

}

object SwingImageViewer {

  def apply(is: Stream[BufferedImage], timerDelay: Int = 1): SwingImageViewer = {
    val siv = new SwingImageViewer(is, timerDelay)
    siv.top.visible = true
    siv
  }

}

case class ImagePanel(var bi: BufferedImage) extends Panel {
  override def paintComponent(g: Graphics2D) = {
    //g.clearRect(0,0,size.width,size.height)
    g.drawImage(bi, 0, 0, null)
  }
}

object ImagePanel {
  def apply(x: Int, y: Int) = {
    val bi = new BufferedImage(x, y, BufferedImage.TYPE_BYTE_BINARY)
    val ip = new ImagePanel(bi)
    ip.preferredSize = new Dimension(x, y)
    ip
  }
}

/* eof */

