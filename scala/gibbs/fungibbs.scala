/*
fungibbs.scala
My Gibbs sampling test code, in scala
Requires ParallelCOLT
Functional version: Look, Ma! No vars! ;-)

scalac fungibbs.scala
time scala FunGibbs > fdata.tab

*/

object FunGibbs {
 
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

    def nextThinnedIter(s: State,left: Int): State = {
       if (left==0) s 
       else nextThinnedIter(nextIter(s),left-1)
    }

    def genIters(s: State,current: Int,stop: Int,thin: Int): State = {
         if (!(current>stop)) {
             println(current+" "+s.x+" "+s.y)
             genIters(nextThinnedIter(s,thin),current+1,stop,thin)
         }
         else s
    }

    def main(args: Array[String]) {
        println("Iter x y")
        genIters(new State(0.0,0.0),1,50000,1000)
     }

}


