/*
Gibbs.scala
My Gibbs sampling test code, in scala
Requires Breeze

time sbt run > gibbs.dat


*/


object Gibbs2 {
 
    import java.util.Date
    import scala.math.sqrt
    import breeze.stats.distributions._
 
    def main(args: Array[String]) {
        val N=50000
        val thin=1000
        var x=0.0
        var y=0.0
        println("Iter x y")
        for (i <- 0 until N) {
            for (j <- 0 until thin) {
                x=new Gamma(3.0,y*y+4).draw
                y=new Gaussian(1.0/(x+1),1.0/sqrt(2*x+2)).draw
            }
            println(i+" "+x+" "+y)
        }
    }
 
}


