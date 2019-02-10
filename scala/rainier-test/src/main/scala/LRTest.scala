/*
Logistic regression test which generates NaNs with noalloc PR
 */

import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import com.stripe.rainier.repl._

object LRTest {

  def main(args: Array[String]): Unit = {

    // first simulate some data from a logistic regression model
    val r = new scala.util.Random(0)
    val N = 1000
    val beta0 = 0.1
    val beta1 = 0.3
    val x = (1 to N) map { i =>
      3.0 * r.nextGaussian
    }
    val theta = x map { xi =>
      beta0 + beta1 * xi
    }
    def expit(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
    val p = theta map expit
    val y = p map (pi => if (r.nextDouble < pi) 1 else 0)

    // now build and fit model
    val model = for {
      beta0 <- Normal(0, 5).param
      beta1 <- Normal(0, 5).param
      _ <- Predictor[Double].from{x =>
          {
            val theta = beta0 + beta1 * x
            val p = Real(1.0) / (Real(1.0) + (Real(0.0) - theta).exp)
            Binomial(p,1)
          }
        }.fit(x zip y)
    } yield (beta0, beta1)

    println("Model built. Running HMC...")
    implicit val rng = ScalaRNG(3)
    val out = model.sample(HMC(5), 10000, 10)
    println("HMC finished.")
    println(out)
  }

}

// eof

