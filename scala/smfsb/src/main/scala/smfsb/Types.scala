/*
Types.scala

 */

package smfsb

object Types {

  import breeze.linalg._

  // Serialisable to a CSV row...
  // TODO: Add "toDvd" to this type class??
  trait CsvRow[T] {
    def toCsv(value: T): String
    }
  implicit class CsvRowSyntax[T](value: T) {
    def toCsv(implicit inst: CsvRow[T]): String = inst.toCsv(value)
  }

  // Parameter type class
  trait Parameter[T] extends CsvRow[T] {
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
  trait State[S] extends CsvRow[S] {
  }
  type IntState = DenseVector[Int]
  implicit val dviState = new State[IntState] {
    def toCsv(s: IntState): String = (s.toArray map (_.toString)).mkString(",")
  }
  type DoubleState = DenseVector[Double]
  implicit val dvdState = new State[DoubleState] {
    def toCsv(s: DoubleState): String = (s.toArray map (_.toString)).mkString(",")
  }

  // Observation type class, with implementations for Ints and Doubles
  trait Observation[O] {
  }
  implicit val dviObs = new Observation[IntState] {
  }
  implicit val dvdObs = new Observation[DoubleState] {
  }

  // Data set type class, for ABC methods
  trait DataSet[D] {
  }
  implicit val tsisDs = new DataSet[Ts[IntState]] {
  }

  // TODO: Make this a type class too...
  type HazardVec = DenseVector[Double]

}

/* eof */

