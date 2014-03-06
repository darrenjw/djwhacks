/*
sim-test.scala

Some basic tests for forwards simulation in the bayeskit package


*/

package bayeskit

import org.scalatest._
import org.scalatest.junit._
import org.scalatest.prop._
import org.junit.runner.RunWith

import breeze.stats.distributions._
import sim._
import lvsim.stepLV
import pfilter._
import pmmh._

@RunWith(classOf[JUnitRunner])
class MyTestSuite extends FunSuite {

  test("1+2=3") {
    assert(1 + 2 === 3)
  }

  test("stepLV") {
    val newState = stepLV(Vector(100, 50), 0, 10, Vector(1.0, 0.005, 0.6))
    assert((newState(0) >= 0) & (newState(1) >= 0))
  }

  test("simTs") {
    val ts = simTs(Vector(100, 50), 0, 100, 0.1, stepLV, Vector(1.0, 0.005, 0.6))
    assert(ts.length >= 1000)
    assert(ts(0)._1 === 0.0)
    assert(ts(0)._2.length === 2)
  }

  test("pfMLLik") {
    def simPrior(n: Int, t: Time, th: Parameter): Vector[State] = {
      val prey = new Poisson(100.0).sample(n).toVector
      val predator = new Poisson(50.0).sample(n).toVector
      prey.zip(predator) map { x => Vector(x._1, x._2) }
    }
    def obsLik(s: State, o: Observation, th: Parameter): Double = {
      new Gaussian(s(0), 10.0).pdf(o(0))
    }
    val truth = simTs(Vector(100, 50), 0, 30, 2.0, stepLV, Vector(1.0, 0.005, 0.6))
    val data = truth map { x => (x._1, Vector(x._2(0).toDouble)) }
    val mll = pfMLLik(100, simPrior, 0.0, stepLV, obsLik, data)
    val mllSample = mll(Vector(1.0, 0.005, 0.6))
    assert(mllSample <= 0.0)
  }

  test("pfMLLikPar") {
    def simPrior(n: Int, t: Time, th: Parameter): Vector[State] = {
      val prey = new Poisson(100.0).sample(n).toVector
      val predator = new Poisson(50.0).sample(n).toVector
      prey.zip(predator) map { x => Vector(x._1, x._2) }
    }
    def obsLik(s: State, o: Observation, th: Parameter): Double = {
      new Gaussian(s(0), 10.0).pdf(o(0))
    }
    val truth = simTs(Vector(100, 50), 0, 30, 2.0, stepLV, Vector(1.0, 0.005, 0.6))
    val data = truth map { x => (x._1, Vector(x._2(0).toDouble)) }
    val mll = pfMLLikPar(100, simPrior, 0.0, stepLV, obsLik, data)
    val mllSample = mll(Vector(1.0, 0.005, 0.6))
    assert(mllSample <= 0.0)
  }

  test("pmmh") {
    def simPrior(n: Int, t: Time, th: Parameter): Vector[State] = {
      val prey = new Poisson(100.0).sample(n).toVector
      val predator = new Poisson(50.0).sample(n).toVector
      prey.zip(predator) map { x => Vector(x._1, x._2) }
    }
    def obsLik(s: State, o: Observation, th: Parameter): Double = {
      new Gaussian(s(0), 10.0).pdf(o(0))
    }
    val truth = simTs(Vector(100, 50), 0, 30, 2.0, stepLV, Vector(1.0, 0.005, 0.6))
    val data = truth map { x => (x._1, Vector(x._2(0).toDouble)) }
    val mll = pfMLLik(100, simPrior, 0.0, stepLV, obsLik, data)
    val pmmhOutput=runPmmh(10,Vector(1.0, 0.005, 0.6),mll)
    assert(pmmhOutput.length==10)
  }

}





/* eof */


