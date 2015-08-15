/*
pois.scala

*/


import org.ddahl.rscala.callback._
import breeze.stats.distributions._
import breeze.linalg._

object ScalaToRTest {

  def main(args: Array[String]) = {

    // first simulate some data consistent with a Poisson regression model
    val x = Uniform(50,60).sample(1000)
    val eta = x map { xi => (xi * 0.1) - 3 }
    val mu = eta map { math.exp(_) }
    val y = mu map { Poisson(_).draw }
    
    // call to R to fit the Poission regression model
    val R = RClient() // initialise an R interpreter
    R.x=x.toArray // send x to R
    R.y=y.toArray // send y to R
    R.eval("mod <- glm(y~x,family=poisson())") // fit the model in R
    // pull the fitted coefficents back into scala
    val beta = DenseVector[Double](R.evalD1("mod$coefficients"))

    // print the fitted coefficents
    println(beta)

  }

}

// eof


