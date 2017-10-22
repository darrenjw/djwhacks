package agglom

import scala.swing._
import scala.swing.event._
import java.awt.{ Graphics2D, Color, BasicStroke }
import java.awt.image.{ BufferedImage, WritableRaster }
import scala.math._
import scala.util.Random
import scala.annotation.tailrec

object MySwingApp extends SimpleSwingApplication {

  val width = 900
  val height = 700
  val r = Random

  def top = new MainFrame {
    title = "Agglomeration demo"
    val onButton = new Button {
      text = "Start"
    }
    val offButton = new Button {
      text = "Stop"
    }
    val panel = ImagePanel(width, height)
    contents = new BoxPanel(Orientation.Vertical) {
      contents += onButton
      contents += offButton
      contents += panel
      border = Swing.EmptyBorder(30, 30, 10, 30)
    }
    listenTo(onButton)
    listenTo(offButton)
    var index=0
    val timer = new javax.swing.Timer(1, Swing.ActionListener(e =>
      {
        agglom(width / 2, 0, 10, panel,index)
        index+=10
        panel.repaint()
      }))
    reactions += {
      case ButtonClicked(b) => {
        if (b.text == "Start")
          timer.start()
        else
          timer.stop()
      }
    }
  }

  @tailrec def agglom(x0: Int, y0: Int, n: Int, panel: ImagePanel, index: Int): Unit = {
    def clash(x: Int, y: Int): Boolean = {
      val byte = panel.bi.getRGB(x, y)
      (byte != new Color(255,255,255).getRGB)
    }
    @tailrec def wander(x: Int, y: Int): (Int, Int) = {
      val u = r.nextDouble
      val xp = if (u < 0.5) { x - 1 } else { x + 1 }
      val yp = if ((u > 0.25) & (u < 0.75)) { y - 1 } else { y + 1 }
      val xn = min(max(xp, 0), width - 1)
      val yn = min(max(yp, 0), height - 1)
      if (clash(xn, yn)) {
        panel.bi.setRGB(x,y,new Color(0,(index/20) % 256,(255-index/20).abs % 256).getRGB)
        (x, y)
      } else {
        wander(xn, yn)
      }
    }
    if (n > 0) {
      wander(x0, y0)
      agglom(x0, y0, n - 1, panel,index+1)
    }
  }

}

object ImagePanel {
  def apply(x: Int, y: Int) = {
    val bi = new BufferedImage(x, y, BufferedImage.TYPE_INT_RGB)
    val wr = bi.getRaster()
    val big = bi.createGraphics()
    val ip = new ImagePanel(bi)
    ip.preferredSize = new Dimension(x, y)
    big.setColor(Color.white)
    big.fillRect(0, 0, x, y)
    wr.setSample(x / 2, y / 2, 0, 0) // seed the agglomeration with a single pixel in the centre
    //wr.setSample(x / 2, y -1, 0, 0) // seed the agglomeration with a single pixel at the base
    big.setColor(Color.black)
    big.drawLine(0,y-1,x,y-1) // seed with a line at the base
    ip
  }
}

class ImagePanel(bi_f: BufferedImage) extends Panel {

  val bi=bi_f
  
  override def paintComponent(g: Graphics2D) = {
    g.drawImage(bi, 0, 0, null)
  }

}




