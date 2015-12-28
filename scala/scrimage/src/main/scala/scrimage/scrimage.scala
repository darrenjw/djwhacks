/*

 */

import java.io.File
import com.sksamuel.scrimage.Image
import com.sksamuel.scrimage.nio.JpegWriter

object ScrimageTest {

  def main(args: Array[String]): Unit = {

    println("Hello")
    Image.fromFile(new File("image.jpg")).flipX.output(new File("out.jpg"))(JpegWriter())

  }

}

// eof

