/*
RWMH.scala

Simple RW MH algorithm for a 
logistic regression model, applied to the Pima data

*/

import smile.data.pimpDataFrame
import annotation.tailrec

import torch.Device.{CPU, CUDA}
import torch.*

type TD = Tensor[Float64]



// a trait for stream-like things that can be thinned
trait Thinnable[F[_]]:
  extension [T](ft: F[T])
    def thin(th: Int): F[T]

// a thinnable instance for LazyList
given Thinnable[LazyList] with
  extension [T](s: LazyList[T])
    def thin(th: Int): LazyList[T] =
      val ss = s.drop(th-1)
      if (ss.isEmpty) LazyList.empty else
        ss.head #:: ss.tail.thin(th)





def mhKernel[S](
    logPost: S => Double, rprop: S => S,
    dprop: (S, S) => Double = (n: S, o: S) => 1.0
  ): ((S, Double)) => (S, Double) =
    state =>
      val (x0, ll0) = state
      val x = rprop(x0)
      val ll = logPost(x)
      val a = ll - ll0 + dprop(x0, x) - dprop(x, x0)
      if (math.log(torch.rand(Seq(1)).item) < a)
        (x, ll)
      else
        (x0, ll0)


object RWMHApp:

  @main def rwmh() =
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
    def ll(beta: TD): Double =
      ((ones + (((y*2 - ones)*matmul(X, beta))*(-1)).exp).log*(-1)).sum.item
    def gll(beta: TD): TD =
      matmul(X.t, (y - ones / (ones + (matmul(X, beta)*(-1)).exp)))

    println("Now define a function for gradient ascent")
    def oneStep(learningRate: Double)(b0: TD): TD =
      b0 + gll(b0)*learningRate
    def ascend(step: TD => TD, init: TD,
        maxIts: Int = 10000, tol: Double = 1e-8, verb: Boolean = true): TD =
      @tailrec def go(b0: TD, ll0: Double, itsLeft: Int): TD =
        if (verb)
          println(s"$itsLeft : $ll0")
        val b1 = step(b0)
        val ll1 = ll(b1)
        if ((math.abs(ll0 - ll1) < tol)|(itsLeft < 1))
          b1
        else
          go(b1, ll1, itsLeft - 1)
      go(init, ll(init), maxIts)


    println("Now run a simple gradient ascent algorithm")
    // Better choose a reasonable init as gradient ascent is terrible...
    val init = Tensor(Seq(-9.8, 0.1, 0, 0, 0, 0, 1.8, 0))
    val opt = ascend(oneStep(1e-6), init)
    println("Inits: " + init)
    println("Init ll: " + ll(init))
    println("Opt: " + opt)
    println("Opt ll: " + ll(opt))


    println("Now RWMH MCMC...")
    val pre = Tensor(Seq(10.0,1.0,1.0,1.0,1.0,1.0,5.0,1.0))
    def rprop(beta: TD): TD =
      ( beta + pre * torch.randn(Seq(p)) * 0.02 ).clone().detach()
    val kern = mhKernel(ll, rprop)
    val kernd = (p: (TD, Double)) =>
      val kp = kern(p)
      ( kp._1.clone().detach(), kp._2)
    val s = LazyList.iterate((opt, Double.NegativeInfinity))(kernd) map (_._1)
    val out = s.drop(150).thin(200).take(10000)
    val os = new java.io.FileWriter("lr-rwmh.csv")
    os.write("V1,V2,V3,V4,V5,V6,V7,V8\n")
    out.foreach(it =>
      os.write(it.toArray.toList.mkString(","))
      os.write("\n")
    )
    os.close()
    println("Goodbye.")

