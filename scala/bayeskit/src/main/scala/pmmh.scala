package bayeskit

object pmmh {

  import scala.annotation.tailrec
  import scala.math.log
  import breeze.stats.distributions._
  import sim._
  import java.io.{ File, Writer, PrintWriter, OutputStreamWriter }


  def runPmmh[P <: Parameter](s: Writer, iters: Int, initialState: P, mll: P => Option[LogLik], peturb: P => P): List[P] = {
    @tailrec def pmmhAcc(itsLeft: Int, currentState: P, currentMll: LogLik, allIts: List[P]): List[P] = {
      System.err.print(itsLeft.toString + " ")
      s.write(currentState.toString + "\n")
      if (itsLeft == 0) allIts else {
        val prop = peturb(currentState)
        val propMll = mll(prop).getOrElse(-1e99) // TODO: not nice - use pattern matching or flatmap?
        if (log(Uniform(0, 1).draw) < propMll - currentMll) {
          pmmhAcc(itsLeft - 1, prop, propMll, prop :: allIts)
        } else {
          pmmhAcc(itsLeft - 1, currentState, currentMll, currentState :: allIts)
        }
      }
    }
    pmmhAcc(iters, initialState, (-1e99).toDouble, Nil).reverse
  }

  def runPmmhPath[P <: Parameter](s: Writer, iters: Int, initialState: P, mll: P => Option[(LogLik, List[State])], peturb: P => P): List[P] = {
    @tailrec def pmmhAcc(itsLeft: Int, currentState: P, currentMll: LogLik, currentPath: List[State], allIts: List[P]): List[P] = {
      System.err.print(itsLeft.toString + " ")
      s.write(currentState.toString + ",")
      s.write((currentPath map { _.toString }).mkString(",") + "\n")
      if (itsLeft == 0) allIts else {
        val prop = peturb(currentState)
        val (propMll, propPath) = mll(prop).getOrElse((-1e99, Nil)) // TODO: not nice - use pattern matching or flatmap?
        if (log(Uniform(0, 1).draw) < propMll - currentMll) {
          pmmhAcc(itsLeft - 1, prop, propMll, propPath, prop :: allIts)
        } else {
          pmmhAcc(itsLeft - 1, currentState, currentMll, currentPath, currentState :: allIts)
        }
      }
    }
    pmmhAcc(iters, initialState, (-1e99).toDouble, mll(initialState).getOrElse((-1e99, Nil))._2, Nil).reverse // this could fail on 1st iter!!!
  }

}