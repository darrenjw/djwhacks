/*
ML-GA-AD.scala

Simple gradient ascent algorithm for maximum likelihood estimation
of a logistic regression model, applied to the Pima data

*/

import smile.data.pimpDataFrame
import annotation.tailrec

import torch.Device.{CPU, CUDA}
import torch.*

object GradientAscentAD:

  @main def runAd() =
    val device = if torch.cuda.isAvailable then CUDA else CPU
    println("First read and process the data")
    val df = smile.read.csv("pima.data", delimiter=" ", header=false)
    print(df)
    val y = Tensor(df.select("V8").
      map(_(0).asInstanceOf[String]).
      map(s => if (s == "Yes") 1.0 else 0.0).toArray.toIndexedSeq)
    println(y)
    val x = Tensor(df.drop("V8").toMatrix.toArray.flatten.toIndexedSeq).
      reshape(df.nrow, df.ncol - 1)
    println(x)
    val ones = onesLike(y)
    val X = cat(Seq(ones.reshape(df.nrow, 1), x), dim=1)
    println(X)
    val p = X.shape(1)
    println(p)

    println("Now define log likelihood and gradient")
    def ll(beta: TD): TD =
      ((ones + (((y*2 - ones)*(X matmul beta))*(-1)).exp).log*(-1)).sum
    def gll(beta: TD): TD =
      beta.requiresGrad = true
      val ll0 = ll(beta)
      ll0.requiresGrad = true
      ll0.backward()
      val g = beta.grad
      g match
        case gd: TD => gd
        case _ =>
          println("Error - no gradient")
          zerosLike(beta)

    println("Now define a function for gradient ascent")
    def oneStep(learningRate: Double)(b0: TD): TD =
      (b0 + gll(b0)*learningRate).detach()
    def ascend(step: TD => TD, init: TD,
        maxIts: Int = 10000, tol: Double = 1e-8, verb: Boolean = true): TD =
      @tailrec def go(b0: TD, ll0: TD, itsLeft: Int): TD =
        if (verb)
          println(s"$itsLeft : $ll0")
          println(s"$b0")
        val b1 = step(b0)
        val ll1 = ll(b1)
        if ((math.abs((ll0 - ll1).item) < tol)|(itsLeft < 1))
          b1
        else
          go(b1, ll1, itsLeft - 1)
      go(init, ll(init), maxIts)

    println("Now run a simple gradient ascent algorithm")
    // Better choose a reasonable init as gradient ascent is terrible...
    val init = Tensor(Seq(-9.8, 0.1, 0, 0, 0, 0, 1.8, 0), requiresGrad=true)
    println(init)
    //val ll0 = ll(init)
    val ll0 = ((ones + (((y*2 - ones)*(X matmul init))*(-1)).exp).log*(-1)).sum
    ll0.requiresGrad = true
    println(ll0)
    println("here0")
    ll0.backward()
    println("here1")
    println(init.grad)
    println("Optimising now...")
    val opt = ascend(oneStep(1e-6), init)
    println("Inits: " + init)
    println("Init ll: " + ll(init))
    println("Opt: " + opt)
    println("Opt ll: " + ll(opt))
    println("Goodbye.")

