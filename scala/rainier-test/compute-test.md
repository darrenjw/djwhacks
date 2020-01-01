Figuring out how the compute graph works...

```scala
import com.stripe.rainier.compute._
// Create a function of three variables
// Example from my APTS notes (p.42 of main notes):
// https://www.staff.ncl.ac.uk/d.j.wilkinson/teaching/apts-sc/
// which is in fact from Nocedal and Wright (2006)...
val x0 = Real.variable()
val x1 = Real.variable()
val x2 = Real.variable()
val y = ( x0*x1*(x2.sin) + (x0*x1).exp )/x2
println(y)
// evaluate using an evaluator (slow)
val eval = new Evaluator(Map(x0 -> 1.0, x1 -> 2.0, x2 -> math.Pi/2.0))
val ey = eval.toDouble(y)
println(ey)
// compile the function for fast evaluation
val cy = Compiler.default.compile(List(x0, x1, x2), y)
val eyc = cy(Array(1.0, 2.0, math.Pi/2.0)) // fast
println(eyc)
// gradients
println(y.gradient.map(eval.toDouble(_))) // WRONG?! BUG?! *******
// compiled gradients
val cg = Compiler.withGradient("y", y, List(x0, x1, x2))
// have gradient functions, but not actually compiled?!
val cg0 = cg.head // function
val cgt = cg.tail // gradients
println(eval.toDouble(cg0._2)) // slow evaluation?
println(cgt.map(e => eval.toDouble(e._2))) // slow evaluation (but correct)?
// now compile the gradient functions?
val cg0c = Compiler.default.compile(List(x0, x1, x2), cg0._2)
println(cg0c(Array(1.0, 2.0, math.Pi/2.0))) // fast
val cgtc = cgt.map(e => Compiler.default.compile(List(x0, x1, x2), e._2))
println(cgtc.map(_(Array(1.0, 2.0, math.Pi/2.0)))) // fast (and correct)
```


#### eof
