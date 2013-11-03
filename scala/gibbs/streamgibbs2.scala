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

    def nextThinnedIter(s: State,left: Int): State =
        if (left==0) s else nextThinnedIter(nextIter(s),left-1)
   
    def mcmcStream(it: Int, s: State, thin: Int): Stream[(Int,State)] = 
        (it,s) #:: mcmcStream(it+1,nextThinnedIter(s,thin),thin)

    def genIters(s: State,current: Int,stop: Int,thin: Int): Stream[(Int,State)] =
        mcmcStream(1,s,thin) take stop

    def main(args: Array[String]) {
        println("Iter x y")
        val its = genIters(new State(0.0,0.0),1,50000,1000).toList
	for (it <- its) yield println(it._1+" "+it._2.x+" "+it._2.y)
    }

}


