/*
sim-test.scala

Some basic tests for forwards simulation in the bayeskit package


*/

package bayeskit

import org.scalatest._
import org.scalatest.junit._
import org.scalatest.prop._
import org.junit.runner.RunWith

@RunWith(classOf[JUnitRunner])
class MyTestSuite extends FunSuite {

  test("1+2=3") {
    assert(1 + 2 === 3)
  }

}





/* eof */


