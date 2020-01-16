If my understanding of how gradient evaluation works is correct (and it may not be), I think there may be a bug in the evaluation of gradients (without compilation), apparently leading to switched gradient components in some cases. A minimal reproducible example/test is given below, for the project `rainierCore` on the current head of the `develop` branch, though the problem also affects the current release.

```scala
import com.stripe.rainier.compute._
val x0 = Real.variable()
val x1 = Real.variable()
val x2 = Real.variable()
val y = ( x0*x1*(x2.sin) + (x0*x1).exp )/x2
val eval = new Evaluator(Map(x0 -> 1.0, x1 -> 2.0, x2 -> math.Pi/2.0))
val egd = y.gradient.map(eval.toDouble(_)) // Incorrect - first two elements switched
val cg = Compiler.withGradient("y", y, List(x0, x1, x2))
val cgd = cg.tail.map(e => eval.toDouble(e._2)) // Correct
val ssd = (egd zip cgd).map(p => p._1 - p._2).map(x => x*x).sum
assert(ssd < 0.001)
```

Both `egd` and `cdg` should contain grad `y`. `cgd` is correct, but `egd` has the first two elements switched.

A version of the above with more explanation can be found on the gitter channel (posted 1st January 2020).

