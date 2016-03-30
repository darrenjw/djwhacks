/*
Types.scala

 */

package smfsb

object Types {

  import breeze.linalg._

  // Parameter type class
  trait Parameter[T] {
    def perturb(value: T): T
    def toCsv(value: T): String
  }
  implicit class ParameterSyntax[T](value: T) {
    def perturb(implicit inst: Parameter[T]): T = inst.perturb(value)
    def toCsv(implicit inst: Parameter[T]): String = inst.toCsv(value)
  }
  // Leave implementations to be model-specific...


  // Hard-coded types...
  type Time = Double
  type Ts[S] = List[(Time, S)]
  type LogLik = Double

  // State type class, with implementations for Ints and Doubles
  trait State[S] {
  }
  type IntState = DenseVector[Int]
  implicit val dviState = new State[IntState] {
  }
  type DoubleState = DenseVector[Double]
  implicit val dvdState = new State[DoubleState] {
  }


  // State type class, with implementations for Ints and Doubles
  trait Observation[O] {
  }
  implicit val dviObs = new Observation[IntState] {
  }
  implicit val dvdObs = new Observation[DoubleState] {
  }


  // TODO: Make this a type class too...
  type HazardVec = DenseVector[Double]

}

/* eof */

