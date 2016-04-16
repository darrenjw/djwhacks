# FP Basics

## Exercise 2

### Part A

Copy your previous `findRoot` function from Exercise 1, and add a new function `findRootOpt` which wraps it, so that instead of returning a `Double` it returns `Option[Double]`. The new signature is:

```scala
findRootOpt(low: Double, high: Double)(f: Double => Double): Option[Double]
```

Add checks that `low < high` and that the sign of `f(low)` is different from the sign of `f(high)` and return `None` if either check fails. Otherwise your function should behave as previously, returning the root in a `Some`.

All of the previous test case translate obviously as follows:

```scala
findRootOpt(-10.0,10.0)(x => x+1.0) == Some(-1.0)

findRootOpt(-5.0,10.0)(x => 2.0-x) == Some(2.0)

findRootOpt(0.0,5.0)(x => x-1.0) == Some(1.0)

findRootOpt(0.0,2.0)(x => (x+1.0)*(x-1.0)) == Some(1.0)

findRootOpt(-2.0,0.0)(x => (x+1.0)*(x-1.0)) == Some(-1.0)

findRootOpt(0.0,2.0)(x => x*x-2.0) == Some(math.sqrt(2.0))
```

In addition, we can add some new test cases which test the inital assumptions:

```scala
findRootOpt(2.0,0.0)(x => x-1.0) == None

findRootOpt(-1.0,-3.0)(x => x+2.0) == None

findRootOpt(0.0,2.0)(x => x+1.0) == None

findRootOpt(0.0,2.0)(x => x-5.0) == None

```

Again, these test cases are all included in the associated Scala template in this directory, and can be run with the `~testOnly PartA` task in `sbt`.


### Part B (if time permits)

(solve a triangular system of nonlinear equations)


You can run all tests for Part A and Part B with the `~test` task in `sbt`, or just the specific tests for Part B with `~testOnly PartB`.

