/*
compute.scala

Using rainier-compute as an auto-diff framework

 */

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import com.stripe.rainier.compute.*

object Compute:

  def compile(expr: Real, params: Set[Real]) =
    val fun = Compiler.default.compileTargets(TargetGroup(List(expr), params))
    val nVars = params.size
    val inputs = new Array[Double](fun.numInputs)
    val globals = new Array[Double](fun.numGlobals)
    val outputs = new Array[Double](fun.numOutputs)
    def evaluate(vars: Array[Double]) =
      System.arraycopy(vars, 0, inputs, 0, nVars)
      fun(inputs, globals, outputs)
      (outputs.head, outputs.tail)
    (vars: Array[Double]) => evaluate(vars)
  

object ComputeDemo extends IOApp.Simple:

  import Compute.compile

  // Example from Nocedal and Wright (2006), p.205, section 8.2.
  // Section 4.3 of my APTS notes
  val x0 = Real.parameter()
  val x1 = Real.parameter()
  val x2 = Real.parameter()
  // derivatives will be output in the order that parameters are created
  val y = ( x0*x1*(x2.sin) + (x0*x1).exp )/x2
  val eval = compile(y,Set(x0,x1,x2))

  val in1 = Array(1.0, 2.0, math.Pi/2.0)
  val in2 = Array(4.0, 4.0, 4.0)

  val out1 = eval(in1)
  val out2 = eval(in2)
  val out3 = eval(in1)
  val out4 = eval(in2)
  val out5 = eval(in1)
  val out6 = eval(in2)

  def run = for
    _ <- IO.println("hello")
    _ <- IO.println(out1._1, out1._2.toList)
    _ <- IO.println(out2._1, out2._2.toList)
    _ <- IO.println(out3._1, out3._2.toList)
    _ <- IO.println(out4._1, out4._2.toList)
    _ <- IO.println(out5._1, out5._2.toList)
    _ <- IO.println(out6._1, out6._2.toList)
    _ <- IO.println("goodbye")
  yield ()




// eof

