/*
ComputeGraph.scala

Messing about with the compute graph

 */

import com.stripe.rainier.compute._

object ComputeGraph {

  def main(args: Array[String]): Unit = {
    println("Hi")
    // Create a function of three variables
    // Example from my APTS notes, p.42 of main notes:
    // https://www.staff.ncl.ac.uk/d.j.wilkinson/teaching/apts-sc/
    // which is in fact from Nocedal and Wright (2006)...
    val x0 = new Variable // Real.variable() in future version of Rainier
    val x1 = new Variable
    val x2 = new Variable
    val y = ( x0*x1*(x2.sin) + (x0*x1).exp )/x2
    println(y)
    // evaluate using an evaluator (slow)
    val eval = new Evaluator(Map(x0 -> 1.0, x1 -> 2.0, x2 -> math.Pi/2.0))
    val ey = eval.toDouble(y)
    println(ey)
    // compile the function for fast evaluation
    val input = Array(1.0, 2.0, math.Pi/2.0)
    val cy = Compiler.default.compile(List(x0, x1, x2), y)
    val eyc = cy(input) // fast
    println(eyc)
    // gradients
    println(y.gradient.map(eval.toDouble(_))) // ***ordered by y.variables!***
    // compiled gradients
    val cg = Compiler.withGradient("y", y, List(x0, x1, x2))
    // have gradient functions, but not actually compiled?!
    val cg0 = cg.head // function
    val cgt = cg.tail // gradients
    println(eval.toDouble(cg0._2)) // slow evaluation?
    println(cgt.map(e => eval.toDouble(e._2))) // slow evaluation?
    // now compile the gradient functions?
    val cg0c = Compiler.default.compile(List(x0, x1, x2), cg0._2)
    println(cg0c(input)) // fast?
    val cgtc = cgt.map(e => Compiler.default.compile(List(x0, x1, x2), e._2))
    println(cgtc.map(_(input))) // fast?
    // The above kind-of works, but you are supposed to do it more like:
    println("Now try the efficient API")
    val cf = Compiler.default.compile(List(x0, x1, x2),
      Compiler.withGradient("y", y, List(x0, x1, x2)))
    val globalBuf = new Array[Double](cf.numGlobals)
    def evaluate(input: Array[Double]): (Double, Array[Double]) = {
      val output = new Array[Double](4)
        (0 to 3).foreach{i =>
          output(i) = cf.output(input, globalBuf, i)
        }
      (output.head, output.tail)
    }
    var out = evaluate(input)
    println(out._1)
    println(out._2.toList)
    out = evaluate(Array(4.0, 4.0, 4.0))
    println(out._1)
    println(out._2.toList)
    out = evaluate(input)
    println(out._1)
    println(out._2.toList)
    out = evaluate(Array(2.0, 1.0, math.Pi/2.0))
    println(out._1)
    println(out._2.toList)
    println("Bye")
  }


}

// eof

