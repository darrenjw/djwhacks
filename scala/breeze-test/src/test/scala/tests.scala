/*
tests.scala

Some basic tests for the breeze library

Mainly testing my understanding of how the library works...

*/

import org.scalatest._
import org.scalatest.junit._
import org.scalatest.prop._
import org.junit.runner.RunWith

import breeze.linalg._

@RunWith(classOf[JUnitRunner])
class MyTestSuite extends FunSuite {

  test("1+2=3") {
    assert(1 + 2 === 3)
  }

  test("int vector") {
    val x = DenseVector.zeros[Int](5)
    assert(x(0) === 0)
    assert(x.length === 5)
  }

  test("matrix mult") {
    val m1 = DenseMatrix.ones[Double](3, 2)
    val m2 = DenseMatrix.ones[Double](2, 3)
    val m3 = m1 * m2
    assert(m3(1, 1) === 2)
    val m4 = m2 * m1
    assert(m4(1, 1) === 3)
  }

}





/* eof */


