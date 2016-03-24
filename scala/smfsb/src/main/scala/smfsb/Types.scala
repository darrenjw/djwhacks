/*
Types.scala

 */

package smfsb

object Types {

  import breeze.linalg._

  trait Parameter

  type Time=Double
  type State=DenseVector[Int]
  type HazardVec=DenseVector[Double]
  type Ts = List[(Time,State)]

}

/* eof */

