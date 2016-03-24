/*
smfsb-test.scala

Some basic tests

*/

package smfsb

import org.scalatest._
import org.scalatest.junit._
import org.scalatest.prop._
import org.junit.runner.RunWith

import breeze.linalg._
import Types._

@RunWith(classOf[JUnitRunner])
class MyTestSuite extends FunSuite {

  test("1+2=3") {
    assert(1 + 2 === 3)
  }

  test("create and step lv model") {
    import SpnExamples._
    val step=stepLv(lvparam)
    val output=step(DenseVector(50,40),0.0,1.0)
    assert(output.length === 2)
}

  test("create and step id model") {
    import SpnExamples._
    val step=stepId(idparam)
    val output=step(DenseVector(0),0.0,1.0)
    assert(output.length === 1)
}

  test("simTs for lv model") {
    import SpnExamples._
    import Sim._
    val step=stepLv(lvparam)
    val ts=simTs(DenseVector(50,40),0.0,20.0,0.1,step)
    plotTs(ts)
    assert(ts.length === 201)
}


}


/* eof */ 

