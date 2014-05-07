

object CSV {

  import breeze.io.CSVReader
  import java.io.FileReader


  def main(args: Array[String]): Unit = {
    println("hello")
    val csv=CSVReader.read(new FileReader("mytest.csv"),skipLines=1)
    println(csv)
    println(csv(1)(1))
    val gender=csv map {x=>x(0)}
    println(gender)
    val height=csv map {x=>x(1)} map {_.toInt}
    println(height)
    val t=height.reduce(_+_)
    val m=t.toDouble/height.length
    println(m)
    println("goodbye")
  }

}
