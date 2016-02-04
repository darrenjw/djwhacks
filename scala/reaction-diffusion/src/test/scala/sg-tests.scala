/*
tests.scala

Some basic tests for the reaction diffusion code


*/

import org.scalatest._
import org.scalatest.junit._
import org.scalatest.prop._
import org.junit.runner.RunWith

import breeze.linalg._

import ReactDiff2dSG._

@RunWith(classOf[JUnitRunner])
class MyReacDiffSuite extends FunSuite {

  test("int vector") {
    val x = DenseVector.zeros[Int](5)
    assert(x(0) === 0)
    assert(x.length === 5)
  }

  test("left-right") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    assert(left(right(m)) === m)
}

  test("right-left") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    assert(right(left(m)) === m)
}

  test("up-down") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    assert(up(down(m)) === m)
}

  test("down-up") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    assert(down(up(m)) === m)
}

  test("left-right-down-up") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    assert(left(right(down(up(m)))) === m)
}

  test("left") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    val ml=new DenseMatrix(4,3,Array[Double](1,2,3,4,2,3,4,5,0,1,2,3))
    assert(left(m) === ml)
}

  test("up") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    val mu=new DenseMatrix(4,3,Array[Double](1,2,3,0,2,3,4,1,3,4,5,2))
    assert(up(m) === mu)
}

  test("rectify") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    assert(rectify(m) === m)
}

  test("rectify2") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    val m2=m.copy
    m2(0,0) = -1.0
    assert(rectify(m2) === m)
}

  test("laplace") {
    val m=new DenseMatrix(3,3,Array[Double](0,1,2,3,4,5,6,7,8))
    val lm=new DenseMatrix(3,3,Array[Double](12,9,6,3,0,-3,-6,-9,-12))
    assert(laplace(m) === lm)
}

  test("sqrt") {
    val m=new DenseMatrix(2,2,Array[Double](1,4,4,9))
    val ms=new DenseMatrix(2,2,Array[Double](1,2,2,3))
    assert(sqrt(m) === ms)
}

  test("elem-times") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    val mm=new DenseMatrix(4,3,Array[Double](0,1,4,9,1,4,9,16,4,9,16,25))
    assert(m :* m === mm)
}

  test("times-const") {
    val m=DenseMatrix.tabulate(4,3){case (i,j) => (i+j).toDouble}
    val m2=DenseMatrix.tabulate(4,3){case (i,j) => ((i+j)*2).toDouble}
    assert(m * 2.0 === m2)
}





}





/* eof */


