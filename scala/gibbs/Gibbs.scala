/*
Gibbs.scala
My Gibbs sampling test code, in scala
Requires ParallelCOLT

scalac GibbsSc.scala
time scala GibbsSc > data.tab

*/


object GibbsSc {
 
    import cern.jet.random.tdouble.engine.DoubleMersenneTwister
    import cern.jet.random.tdouble.Normal
    import cern.jet.random.tdouble.Gamma
    import Math.sqrt
    import java.util.Date
 
    def main(args: Array[String]) {
        val N=50000
        val thin=1000
        val rngEngine=new DoubleMersenneTwister(new Date)
        val rngN=new Normal(0.0,1.0,rngEngine)
        val rngG=new Gamma(1.0,1.0,rngEngine)
        var x=0.0
        var y=0.0
        println("Iter x y")
        for (i <- 0 until N) {
            for (j <- 0 until thin) {
                x=rngG.nextDouble(3.0,y*y+4)
                y=rngN.nextDouble(1.0/(x+1),1.0/sqrt(2*x+2))
            }
            println(i+" "+x+" "+y)
        }
    }
 
}


