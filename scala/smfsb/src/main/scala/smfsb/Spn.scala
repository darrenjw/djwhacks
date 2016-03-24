/*
Spn.scala

Functionality relating to Spn objects

 */

package smfsb

import Types._
import breeze.linalg._

case class Spn[P <: Parameter](
  species: List[String],
  pre: DenseMatrix[Int],
  post: DenseMatrix[Int],
  h: P => (State, Time) => HazardVec
)


object SpnExamples {

  // Lotka-Volterra model
  case class LvParameter(th0: Double, th1: Double, th2: Double) extends Parameter
  val lvparam = LvParameter(1.0, 0.005, 0.6)
  val lv = Spn(
    List("x", "y"),
    DenseMatrix((1, 0), (1, 1), (0, 1)),
    DenseMatrix((2, 0), (0, 2), (0, 0)),
    (th: LvParameter) => (x, t) => DenseVector(x(0) * th.th0, x(0) * x(1) * th.th1, x(1) * th.th2)
  )
  val stepLv=Step.gillespie(lv)

  // Immigration death model
  case class IdParameter(alpha: Double, mu: Double) extends Parameter
  val idparam = IdParameter(1.0, 0.1)
  val id = Spn(
    List("X"),
    DenseMatrix((0), (1)),
    DenseMatrix((1), (0)),
    (th: IdParameter) => (x, t) => DenseVector(th.alpha, x(0) * th.mu)
  )
  val stepId=Step.gillespie(id)

  // Michaelis Menten



  // Auto-regulatory network



}

/* eof */

