import scala.swing._
import scala.swing.event._
import java.awt.{Graphics2D,Color}
import scala.math._
import scala.util.Random

object MySwingApp extends SimpleSwingApplication {

 def top = new MainFrame {
  title="Reactive swing app"
  val button=new Button {
   text="Click me"
  }
  val label=new Label {
   text="No button clicks registered"
  }
  val panel=new Canvas {
   preferredSize=new Dimension(500,500)
  }
  contents=new BoxPanel(Orientation.Vertical) {
   contents+=button
   contents+=label
   contents+=panel
   border=Swing.EmptyBorder(30,30,10,30)
  }
  listenTo(button)
  var nClicks=0
  reactions+={
   case ButtonClicked(b) =>
    nClicks+=1
    label.text="Number of clicks: "+nClicks
    panel.repaint()
  }
 }


}


class Canvas extends Panel {

 override def paintComponent(g: Graphics2D) = {
  g.clearRect(0,0,size.width,size.height)
  g.setColor(Color.blue)
  g.fillOval(0,0,100,100)
  g.drawLine(0,200,100,300)
  g.drawLine(300,300,300,300)
  // g.plotPoint(400,300)
  g.drawRect(100,0,20,20)
  g.fillRect(100,50,20,20)
  g.setFont(new Font("Ariel",java.awt.Font.ITALIC,24))
  g.drawString("Hello",100,100)
  g.fillPolygon(Array(300,350,400),Array(50,0,50),3)
  g.setColor(Color.red)
  sier(250,0,200,200,300,200,g)
  g.setColor(Color.green)
  val r=Random
  tree(250.0,450.0,150.0,-Pi/2,r,g)
 }


 def tree(x: Double,y: Double,l: Double,a: Double,r: Random,g: Graphics2D): Unit = {
  // println(x,y,l,a)
  if (l>2.0) {
   val a1=a+0.3*(r.nextDouble-0.5)
   val x1=x+l*cos(a1)
   val y1=y+l*sin(a1)
   g.drawLine(x.toInt,y.toInt,x1.toInt,y1.toInt)
   val l1=(0.6+0.2*(r.nextDouble-0.5))*l
   tree(x1,y1,l1,a1+Pi/6,r,g)
   val x2=0.3*x+0.7*x1
   val y2=0.3*y+0.7*y1
   val l2=(0.7+0.2*(r.nextDouble-0.5))*l
   tree(x2,y2,l2,a1-Pi/4,r,g)
  }
 }

 def sier(x1: Int,y1: Int,x2: Int,y2: Int,x3: Int,y3: Int,g: Graphics2D): Unit = {
  // println(x1,y1,x2,y2,x3,y3)
  if ((abs(x1-x2)<2)&(abs(x1-x3)<2)&(abs(y1-y2)<2)&(abs(y1-y3)<2))
   g.drawLine(x1,y1,x1,y1)
  else {
   val x12: Int = round(0.5*(x1+x2)).toInt
   val x13: Int = round(0.5*(x1+x3)).toInt
   val x23: Int = round(0.5*(x2+x3)).toInt
   val y12: Int = round(0.5*(y1+y2)).toInt
   val y13: Int = round(0.5*(y1+y3)).toInt
   val y23: Int = round(0.5*(y2+y3)).toInt
   sier(x1,y1,x12,y12,x13,y13,g)
   sier(x12,y12,x2,y2,x23,y23,g)
   sier(x13,y13,x23,y23,x3,y3,g)
  }
 }

}




