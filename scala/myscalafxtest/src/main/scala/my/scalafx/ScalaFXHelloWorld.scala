package my.scalafx

import scala.collection.JavaConverters._
import scalafx.animation.AnimationTimer
import scalafx.application.JFXApp
import scalafx.application.JFXApp.PrimaryStage
import scalafx.scene.Scene
import scalafx.scene.image.{ImageView, WritableImage}
import scalafx.scene.paint.Color._
import scalafx.scene.paint._
import scalafx.scene.text.Text

object ScalaFXHelloWorld extends JFXApp {

  def noisyImage: WritableImage = {
    val im = new WritableImage(500, 400)
    val pw = im.pixelWriter
    // pw.setColor(200,100,Color.rgb(200,100,100))
    //pw.setColor(200,100,Color.gray(1))
    (0 until 500) foreach (i =>
      (0 until 400) foreach (j =>
        pw.setColor(i, j, Color.gray(math.random()))
        )
      )
    im
  }

  stage = new PrimaryStage {
    title = "My ScalaFX test"
    scene = new Scene {
      fill = Color.rgb(0,0,0)
      val iv = new ImageView(noisyImage)
      content = iv
      val delay = 1e9 // nano seconds
      var lastUpdate=0L
      val timer = AnimationTimer(t => {
        if (t-lastUpdate > delay) {
          iv.image = noisyImage
          lastUpdate = t
        }
      })
      timer.start
    }

  }
}
