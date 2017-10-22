/*
shoving-test.scala

Some basic tests for shoving code

*/

package shoving

import org.scalatest._
import org.scalatest.junit._
import org.scalatest.prop._
import org.junit.runner.RunWith

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics._

@RunWith(classOf[JUnitRunner])
class ShovingTest extends FunSuite {

  // Code for testing approximate equality of Doubles
  val eps = 0.00001
  def approxeq(x: Double, y: Double): Boolean = abs(x - y) < eps

  // For some reason I always include this test...   ;-)
  test("1+2=3") {
    assert(1 + 2 === 3)
  }

}





/* eof */


