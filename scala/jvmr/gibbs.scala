/*
gibbs.scala

*/

package gibbs

object Gibbs {

    import scala.annotation.tailrec
    import scala.math.sqrt
    import breeze.stats.distributions.{Gamma,Gaussian}

    case class State(x: Double, y: Double) {
      override def toString: String = x.toString + " , " + y + "\n"
    }

    def nextIter(s: State): State = {
      val newX = Gamma(3.0, 1.0/((s.y)*(s.y)+4.0)).draw
      State(newX, Gaussian(1.0/(newX+1), 1.0/sqrt(2*newX+2)).draw)
    }

    @tailrec def nextThinnedIter(s: State,left: Int): State =
      if (left==0) s else nextThinnedIter(nextIter(s),left-1)

    def genIters(s: State, stop: Int, thin: Int): List[State] = {
      @tailrec def go(s: State, left: Int, acc: List[State]): List[State] =
        if (left>0)
          go(nextThinnedIter(s,thin), left-1, s::acc)
          else acc
      go(s,stop,Nil).reverse
    }

    def main(args: Array[String]) = {
      val out = genIters(State(0.0,0.0),10000,100)
      val s = new java.io.FileWriter("output.csv")
      s.write("x , y\n")
      out map { it => s.write(it.toString) }
      s.close
    }

}


