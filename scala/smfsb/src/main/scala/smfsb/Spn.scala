/*
Spn.scala

Functionality relating to Spn objects

 */

package smfsb

import Types._
import breeze.linalg._
import breeze.stats.distributions.Gaussian
import math.exp

case class Spn[P: Parameter, S: State](
  species: List[String],
  pre: DenseMatrix[Int],
  post: DenseMatrix[Int],
  h: P => (S, Time) => HazardVec
)

object SpnExamples {

  // Lotka-Volterra model 
  case class LvParameter(th0: Double, th1: Double, th2: Double)
  implicit val lvParameter = new Parameter[LvParameter] {
    def perturb(p: LvParameter) = LvParameter(p.th0 * exp(Gaussian(0, 0.05).draw), p.th1 * exp(Gaussian(0, 0.05).draw), p.th2 * exp(Gaussian(0, 0.05).draw))
    def toCsv(p: LvParameter) = "" + p.th0 + "," + p.th1 + "," + p.th2
    def toDvd(p: LvParameter) = DenseVector(p.th0,p.th1,p.th2)
  }
  val lvparam = LvParameter(1.0, 0.005, 0.6)
  val lv = Spn[LvParameter, IntState](
    List("x", "y"),
    DenseMatrix((1, 0), (1, 1), (0, 1)),
    DenseMatrix((2, 0), (0, 2), (0, 0)),
    (th: LvParameter) => (x, t) => DenseVector(
      x(0) * th.th0,
      x(0) * x(1) * th.th1, x(1) * th.th2
    )
  )
  val stepLv = Step.gillespie(lv)
  val stepLvPts = Step.pts(lv)

  // Immigration death model
  case class IdParameter(alpha: Double, mu: Double)
  implicit val idParameter = new Parameter[IdParameter] {
    def perturb(p: IdParameter) = p
    def toCsv(p: IdParameter) = p.toString
    def toDvd(p: IdParameter) = DenseVector(p.alpha,p.mu)
  }
  val idparam = IdParameter(1.0, 0.1)
  val id = Spn[IdParameter, IntState](
    List("X"),
    DenseMatrix((0), (1)),
    DenseMatrix((1), (0)),
    (th: IdParameter) => (x, t) => DenseVector(th.alpha, x(0) * th.mu)
  )
  val stepId = Step.gillespie(id)

  // Michaelis Menten
  case class MmParameter(c1: Double, c2: Double, c3: Double)
  implicit val mmParameter = new Parameter[MmParameter] {
    def perturb(value: MmParameter) = value
    def toCsv(p: MmParameter) = p.toString
    def toDvd(p: MmParameter) = DenseVector(p.c1,p.c2,p.c3)
  }
  val mmparam = MmParameter(0.00166, 1e-04, 0.1)
  val mm = Spn[MmParameter, IntState](
    List("S", "E", "SE", "P"),
    DenseMatrix((1, 1, 0, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
    DenseMatrix((0, 0, 1, 0), (1, 1, 0, 0), (0, 1, 0, 1)),
    (th: MmParameter) => (x, t) => DenseVector(th.c1 * x(0) * x(1), th.c2 * x(2), th.c3 * x(2))
  )
  val stepMm = Step.gillespie(mm)

  // Auto-regulatory network
  case class ArParameter(c: DenseVector[Double])
  implicit val arParameter = new Parameter[ArParameter] {
    def perturb(value: ArParameter) = ArParameter(value.c.map(_ * math.exp(Gaussian(0.0, 0.1).draw)))
    def toCsv(p: ArParameter) = (p.c.toArray map (_.toString)).mkString(",")
    def toDvd(p: ArParameter) = p.c
  }
  val arparam = ArParameter(DenseVector(1.0, 10.0, 0.01, 10.0, 1.0, 1.0, 0.1, 0.01))
  val ar = Spn[ArParameter, IntState](
    List("g.P2", "g", "r", "P", "P2"),
    DenseMatrix((0, 1, 0, 0, 1), (1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 2, 0), (0, 0, 0, 0, 1), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0)),
    DenseMatrix((1, 0, 0, 0, 0), (0, 1, 0, 0, 1), (0, 1, 1, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 1), (0, 0, 0, 2, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    (th: ArParameter) => (x, t) => DenseVector(th.c(0) * x(1) * x(4), th.c(1) * x(0), th.c(2) * x(1), th.c(3) * x(2), th.c(4) * 0.5 * x(3) * (x(3) - 1), th.c(5) * x(4), th.c(6) * x(2), th.c(7) * x(3))
  )
  val stepAr = Step.gillespie(ar)

  // Continuous (CLE) version of the Lotka-Volterra model
  val lvc = Spn[LvParameter, DoubleState](
    List("x", "y"),
    DenseMatrix((1, 0), (1, 1), (0, 1)),
    DenseMatrix((2, 0), (0, 2), (0, 0)),
    (th: LvParameter) => (x, t) => DenseVector(
      x(0) * th.th0,
      x(0) * x(1) * th.th1, x(1) * th.th2
    )
  )
  val stepLvc = Step.cle(lvc)

}

/* eof */

