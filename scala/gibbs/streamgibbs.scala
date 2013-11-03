/*
streamgibbs.scala
My Gibbs sampling test code, in scala
Requires ParallelCOLT
Functional version, using "stream"s

scalac streamgibbs.scala
time scala StreamGibbs > sdata.tab

*/

object StreamGibbs {
 
    import cern.jet.random.tdouble.engine.DoubleMersenneTwister
    import cern.jet.random.tdouble.Normal
    import cern.jet.random.tdouble.Gamma
    import java.util.Date
    import scala.math.sqrt

    val rngEngine=new DoubleMersenneTwister(new Date)
    val rngN=new Normal(0.0,1.0,rngEngine)
    val rngG=new Gamma(1.0,1.0,rngEngine)

    class State(val x: Double,val y: Double)

    def nextIter(s: State): State = {
         val newX=rngG.nextDouble(3.0,(s.y)*(s.y)+4.0)
         new State(newX, 
              rngN.nextDouble(1.0/(newX+1),1.0/sqrt(2*newX+2)))
    }

    def mcmcStream(it: Int, s: State): Stream[(Int,State)] = 
         (it,s) #:: mcmcStream(it+1,nextIter(s))

    def thinnedStream(s: State, thin: Int): Stream[(Int,State)] = 
         mcmcStream(0,s) filter (p => p._1 % thin == 0)

    def genIters(s: State,current: Int,stop: Int,thin: Int): String = {
         (thinnedStream(s,thin) take stop).toList.toString
    }

    def main(args: Array[String]) {
        println("Iter x y")
        println(genIters(new State(0.0,0.0),1,50000,1000))
     }

}


