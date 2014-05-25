package bayeskit

object pmmh {

  import scala.annotation.tailrec
  import scala.math.exp
  import scala.math.log
  import breeze.stats.distributions._
  import sim._
  import java.io._

  def runPmmh(s: Writer, iters: Int, initialState: Parameter, mll: Parameter => Option[Double]): List[Parameter] = {
    @tailrec
    def pmmhAcc(itsLeft: Int, currentState: Parameter, currentMll: Double, allIts: List[Parameter]): List[Parameter] = {
      System.err.print(itsLeft.toString+" ")
      s.write(currentState.mkString(",")+"\n")
      if (itsLeft == 0) allIts else {
        val prop = currentState map { _ * exp(Gaussian(0, 0.01).draw) }
        val propMll = mll(prop).getOrElse(-1e99) // not nice - use pattern matching or flatmap?
        if (log(Uniform(0, 1).draw) < propMll - currentMll) {
          pmmhAcc(itsLeft - 1, prop, propMll, prop :: allIts)
        } else {
          pmmhAcc(itsLeft - 1, currentState, currentMll, currentState :: allIts)
        }
      }
    }
    pmmhAcc(iters, initialState, (-1e99).toDouble, Nil).reverse
  }

  def runPmmhPath(s: Writer, iters: Int, initialState: Parameter, mll: Parameter => Option[(Double,List[State])]): List[Parameter] = {
    @tailrec
    def pmmhAcc(itsLeft: Int, currentState: Parameter, currentMll: Double, currentPath: List[State], allIts: List[Parameter]): List[Parameter] = {
      System.err.print(itsLeft.toString+" ")
      s.write(currentState.mkString(",")+",")
      s.write((currentPath map {_(0)}).mkString(",")+",")
      s.write((currentPath map {_(1)}).mkString(",")+"\n")
      if (itsLeft == 0) allIts else {
        val prop = currentState map { _ * exp(Gaussian(0, 0.01).draw) }
        val (propMll,propPath) = mll(prop).getOrElse((-1e99,Nil)) // not nice - use pattern matching or flatmap?
        if (log(Uniform(0, 1).draw) < propMll - currentMll) {
          pmmhAcc(itsLeft - 1, prop, propMll, propPath, prop :: allIts)
        } else {
          pmmhAcc(itsLeft - 1, currentState, currentMll, currentPath, currentState :: allIts)
        }
      }
    }
    pmmhAcc(iters, initialState, (-1e99).toDouble, mll(initialState).getOrElse((-1e99,Nil))._2, Nil).reverse // this could fail on 1st iter!!!
  }

}