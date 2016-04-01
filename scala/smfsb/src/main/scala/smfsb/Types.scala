/*
Types.scala

Types and type classes used throughout the package

 */

package smfsb

object Types {

  import breeze.linalg._

  // Hard-coded types:

  type Time = Double
  type Ts[S] = List[(Time, S)]
  type LogLik = Double
  // TODO: Make HazardVec a type class?
  type HazardVec = DenseVector[Double]


  // Now type classes:

  // Serialisable to a CSV row (and numeric vector)...
  trait CsvRow[T] {
    def toCsv(value: T): String
    def toDvd(value: T): DenseVector[Double]
    }
  implicit class CsvRowSyntax[T](value: T) {
    def toCsv(implicit inst: CsvRow[T]): String = inst.toCsv(value)
    def toDvd(implicit inst: CsvRow[T]): DenseVector[Double] = inst.toDvd(value)
  }


  // Parameter type class
  trait Parameter[T] extends CsvRow[T] {
    def perturb(value: T): T
  }
  implicit class ParameterSyntax[T](value: T) {
    def perturb(implicit inst: Parameter[T]): T = inst.perturb(value)
  }
  // No implementations - leave to be model-specific...

  // State type class, with implementations for Ints and Doubles
  trait State[S] extends CsvRow[S] {
  }
  type IntState = DenseVector[Int]
  implicit val dviState = new State[IntState] {
    def toCsv(s: IntState): String = (s.toArray map (_.toString)).mkString(",")
    def toDvd(s: IntState): DenseVector[Double] = s.map(_*1.0)
  }
  type DoubleState = DenseVector[Double]
  implicit val dvdState = new State[DoubleState] {
    def toCsv(s: DoubleState): String = (s.toArray map (_.toString)).mkString(",")
    def toDvd(s: DoubleState): DenseVector[Double] = s
  }


  // Now type classes for inferential methods

  // Observation type class, with implementations for Ints and Doubles
  trait Observation[O] extends CsvRow[O] {
  }
  implicit val dviObs = new Observation[IntState] {
    def toCsv(s: IntState): String = (s.toArray map (_.toString)).mkString(",")
    def toDvd(s: IntState): DenseVector[Double] = s.map(_*1.0)
  }
  implicit val dvdObs = new Observation[DoubleState] {
    def toCsv(s: DoubleState): String = (s.toArray map (_.toString)).mkString(",")
    def toDvd(s: DoubleState): DenseVector[Double] = s
  }

  // Data set type class, for ABC methods
  trait DataSet[D] {
  }
  implicit val tsisDs = new DataSet[Ts[IntState]] {
  }
  implicit val tsdsDs = new DataSet[Ts[DoubleState]] {
  }


}

/* eof */

