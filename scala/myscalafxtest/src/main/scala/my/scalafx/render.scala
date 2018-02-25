package my.scalafx

import javafx.embed.swing.{ JFXPanel, SwingFXUtils }
import scalafx.Includes._
import scalafx.animation.AnimationTimer
import scalafx.application.Platform
import scalafx.geometry.Insets
import scalafx.stage.Stage
import scalafx.scene.Scene
import scalafx.scene.layout.{ HBox, VBox }
import scalafx.scene.control.Button
import scalafx.scene.image.{ ImageView, Image }
import scalafx.scene.paint.Color

case class ScalaView(var is: Stream[Image], timerDelay: Int, autoStart: Boolean = false, var saveFrames: Boolean = false) {

  new JFXPanel() // trick to spin up JFX

  Platform.runLater {
    val mainStage = new Stage {
      title = "ScalaFX Image Viewer"
      scene = new Scene {
        fill = Color.rgb(0, 0, 0)
        val iv = new ImageView(is.head)
        val buttons = new HBox {
          padding = Insets(10,10,10,10)
          children = Seq(
            new Button {
              text = "Start"
              onAction = handle { timer.start }
            },
            new Button {
              text = "Stop"
              onAction = handle { timer.stop }
            },
            new Button {
              text = "Save frames"
              onAction = handle { saveFrames = true }
            },
            new Button {
              text = "Don't save frames"
              onAction = handle { saveFrames = false }
            }
          )
        }
        content = new VBox {
          padding = Insets(10,10,10,10)
          children = Seq(buttons,iv)
        }
        var lastUpdate = 0L
        var frameCounter = 0L
        val timer = AnimationTimer(t => {
          if (t - lastUpdate > timerDelay) {
            iv.image = is.head
            if (saveFrames) { javax.imageio.ImageIO.write(SwingFXUtils.fromFXImage(is.head,null), "png", new java.io.File(f"siv-$frameCounter%06d.png")) }
            is = is.tail
            frameCounter += 1
            lastUpdate = t
          }
        })
        if (autoStart) timer.start
      }
    }
    mainStage.showAndWait()
  }

}
