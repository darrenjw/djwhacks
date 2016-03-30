/*
Abc.scala

Simple routines for basic ABC based Bayesian inference

 */

package smfsb

object Abc {

  import Types._

  def abcDistance[P: Parameter, D: DataSet](model: P => D, distance: D => Double)(th: P): Double = distance(model(th))



}

/* eof */

