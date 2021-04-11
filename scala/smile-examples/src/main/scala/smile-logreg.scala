/*
smile-logreg.scala

 */

object SmileLogReg {

  def main(args: Array[String]): Unit = {
    println("Logistic regression example")

    println("Simulate some synthetic data")
    val rng = scala.util.Random
    val N = 2000
    val beta = (0.1, 0.3)
    val x = Array.fill(N)(rng.nextGaussian())
    val theta = x map (xi => beta._1 + beta._2*xi)
    def expit(x: Double): Double = 1.0/(1.0+math.exp(-x))
    val p = theta map expit
    val y = p map (pi => if (rng.nextDouble() < pi) 1 else 0)

    println("Fit logistic regression model")
    import scala.language.postfixOps
    val mod = smile.classification.logit(x map (Array(_)), y)
    println(mod)
    println(mod.predict(Array(-1.0)))
    println(mod.predict(Array(0.0)))
    println(mod.predict(Array(1.0)))

    println("plot predictions")
    import java.awt.Color
    import smile.plot.show
    import smile.plot.swing._
    import smile.plot.Render._
    show(plot((x zip y).map(xyi => Array(xyi._1,xyi._2)), '*'))
    show(plot(x.map(xi => Array(xi,mod.predict(Array(xi)))), '*'))

  }

}

// eof

