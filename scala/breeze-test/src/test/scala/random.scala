/*
tests.scala

Some basic tests for the breeze library

Mainly testing my understanding of how the library works...

*/

import org.scalatest._
import org.scalatest.junit._
import org.scalatest.prop._
import org.junit.runner.RunWith

import breeze.stats.distributions._

@RunWith(classOf[JUnitRunner])
class MyRandomSuite extends FunSuite {

  test("Poisson sample") {
    val poi = new Poisson(3.0)
    val s = poi.sample(10)
    assert(s.length === 10)
    assert(s(1) >= 0)
  }

}





/* eof */


