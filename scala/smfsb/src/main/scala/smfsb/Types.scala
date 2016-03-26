/*
Types.scala

 */

package smfsb

object Types {

  import breeze.linalg._

  // Parameter type class
  trait Parameter[T] {
    def perturb(value: T): T
  }
  implicit class ParameterSyntax[T](value: T) {
    def perturb(implicit inst: Parameter[T]): T = inst.perturb(value)
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

  // TODO: Make these type classes too...
  type HazardVec = DenseVector[Double]
  type Observation = DenseVector[Double]

}

/* eof */

