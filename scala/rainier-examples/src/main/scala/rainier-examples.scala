/*
rainier-examples.scala

*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import com.stripe.rainier.core.*
import com.stripe.rainier.compute.*
import com.stripe.rainier.sampler.*

object BasicTest:
  val a = Normal(0, 1).latent
  val b = Normal(a, 1).latent

object EggModel:
  val eggs = List[Long](45, 52, 45, 47, 41, 42, 44, 42, 46, 38, 36,
    35, 41, 48, 42, 29, 45, 43, 45, 40, 42, 53, 31, 48, 40, 45, 39,
    29, 45, 42)
  val lambda = Gamma(0.5, 100).latent
  val eggModel = Model.observe(eggs, Poisson(lambda))
  val eggSampler = EHMC(5000, 500) // warmup, its
  val eggSamples = eggModel.sample(eggSampler).thin(2) // 4 chains by default

object ComputeTest:
  // Example from Nocedal and Wright (2006), p.205, section 8.2.
  // Section 4.3 of my APTS notes
  val x0 = Real.parameter()
  val x1 = Real.parameter()
  val x2 = Real.parameter()
  val y = ( x0*x1*(x2.sin) + (x0*x1).exp )/x2
  // evaluate using an evaluator (slow)
  val eval = new Evaluator(Map(x0 -> 1.0, x1 -> 2.0, x2 -> math.Pi/2.0))
  val ey = eval.toDouble(y)
  // compile the function for fast evaluation
  val input = Array(1.0, 2.0, math.Pi/2.0)
  val cy = Compiler.default.compile(List(x0, x1, x2), y)
  val eyc = cy(input) // fast
  // Compiler.default.compileTargets(TargetGroup(likelihoods, track))
  val params = List(x0, x1, x2)
  val targ = Target("y", y, params)
  val grad = targ.gradient.map(Compiler.default.compile(params, _))
  def ev(vars: Array[Double]) = grad.map(_(vars))
  // More efficient method below... but relies on ordering of variables introduced
  val target = TargetGroup(List(y), params.toSet)
  val g = Compiler.default.compileTargets(target)
  val nVars = params.size
  val inputs = new Array[Double](g.numInputs)
  val globals = new Array[Double](g.numGlobals)
  val outputs = new Array[Double](g.numOutputs)
  def evaluate(vars: Array[Double]) =
    System.arraycopy(vars, 0, inputs, 0, nVars)
    g(inputs, globals, outputs)
    (outputs.head, outputs.tail.toList) // unnecessary copying here, for now...

object RainierExamples extends IOApp.Simple:

  import EggModel.{lambda, eggSamples}
  import ComputeTest.{ev, evaluate}

  def run = for
    _ <- IO.println(eggSamples.diagnostics) // Rhat, ESS
    _ <- IO.println(eggSamples.predict(lambda).take(20))
    _ <- IO.println(ComputeTest.y)
    _ <- IO.println(ev(Array(1.0, 2.0, math.Pi/2.0)))
    _ <- IO.println(ev(Array(4, 4, 4)))
    _ <- IO.println(ev(Array(1.0, 2.0, math.Pi/2.0)))
    _ <- IO.println(ev(Array(4, 4, 4)))
    _ <- IO.println(ev(Array(1.0, 2.0, math.Pi/2.0)))
    _ <- IO.println(ev(Array(4, 4, 4)))
    _ <- IO.println(ComputeTest.ey)
    _ <- IO.println(ComputeTest.eyc)
    _ <- IO.println(evaluate(Array(1.0, 2.0, math.Pi/2.0)))
    _ <- IO.println(evaluate(Array(4.0, 4.0, 4.0)))
    _ <- IO.println(evaluate(Array(1.0, 2.0, math.Pi/2.0)))
    _ <- IO.println(evaluate(Array(4.0, 4.0, 4.0)))
    _ <- IO.println(evaluate(Array(1.0, 2.0, math.Pi/2.0)))
    _ <- IO.println(evaluate(Array(4.0, 4.0, 4.0)))
    _ <- IO.println(ComputeTest.params.map(_.param.sym.id)) // variable ordering   
  yield ()
