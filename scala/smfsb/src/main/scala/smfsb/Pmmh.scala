/*
Pmmh.scala

 */

package smfsb

object Pmmh {

  import scala.annotation.tailrec
  import scala.math.log
  import breeze.stats.distributions._
  import java.io.{File, Writer, PrintWriter, OutputStreamWriter}
  import Types._


  // TODO: Re-write this completely in terms of transition kernels and Markov chain simulation

  // TODO: This is just a quick hack to get things running...


  def runPmmh[P: Parameter](s: Writer, iters: Int, initialState: P, mll: P => LogLik): List[P] = {
    @tailrec def go(itsLeft: Int, currentState: P, currentMll: LogLik, allIts: List[P]): List[P] = {
      System.err.print(itsLeft.toString + " ")
      s.write(currentState.toCsv + "\n")
      if (itsLeft == 0) allIts else {
        val prop = currentState.perturb
        val propMll = mll(prop)
        if (log(Uniform(0, 1).draw) < propMll - currentMll) {
          go(itsLeft - 1, prop, propMll, prop :: allIts)
        } else {
          go(itsLeft - 1, currentState, currentMll, currentState :: allIts)
        }
      }
    }
    go(iters, initialState, (-1e99).toDouble, Nil).reverse
  }

}

/* eof */

