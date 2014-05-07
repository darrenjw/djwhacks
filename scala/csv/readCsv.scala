

object CSV {

  import breeze.io.CSVReader
  import java.io.FileReader


  def main(args: Array[String]): Unit = {
    println("hello")
    val csv=CSVReader.read(new FileReader("mytest.csv"),skipLines=1)
    println(csv)
    println(csv(1)(1))
    println("goodbye")
  }

}
