import org.scalatest.flatspec.AnyFlatSpec

class SetSpec extends AnyFlatSpec:

  import breeze.stats.distributions.Rand.VariableSeed.randBasis
  import breeze.stats.distributions.*

  "A Poisson(10.0)" should "have mean 10.0" in {
    val p = Poisson(10.0)
    val m = p.mean
    assert(math.abs(m - 10.0) < 0.000001)
  }

  import DCT.*
  import breeze.linalg.*

  // create a random, random length vector for testing
  val N = 20 + Poisson(20).draw()
  val x = DenseVector(Gaussian(1.0,2.0).sample(N).toArray)
  val tol = 1e-6
  //println(N)
  //println(x)

  "A DCT" should "have the correct length" in {
     val xt = dct0(x)
     assert(xt.length == N)
  }

  "dct" should "agree with dct0" in {
    assert(norm(dct(x) - dct0(x)) < tol)
  }

  "dctj" should "agree with dct0" in {
    assert(norm(dctj(x) - dct0(x)) < tol)
  }

  "idct" should "agree with idct0" in {
    assert(norm(idct(x) - idct0(x)) < tol)
  }

  "idctj" should "agree with idct0" in {
    assert(norm(idctj(x) - idct0(x)) < tol)
  }

  "dct0" should "invert" in {
    assert(norm(idct0(dct0(x)) - x) < tol)
  }

  "dct" should "invert" in {
    assert(norm(idct(dct(x)) - x) < tol)
  }

  "dctj" should "invert" in {
    assert(norm(idctj(dctj(x)) - x) < tol)
  }

  val M = 20 + Poisson(20).draw()
  val Mat = DenseMatrix.fill(N,M)(Gaussian(1.0,2.0).draw())

  def mnorm(m: DenseMatrix[Double]): Double =
    norm(DenseVector(m.toArray))

  "A 2d DCT" should "have the correct dimensions" in {
     val xt = dct2(Mat)
     assert(xt.rows == N)
     assert(xt.cols == M)
  }

  "A 2d DCT" should "invert" in {
     assert(mnorm(dct2(dct2(Mat), true) - Mat) < tol)
  }

  "dct20" should "agree with dct2" in {
    assert(mnorm(dct20(Mat) - dct2(Mat)) < tol)
  }

  "idct20" should "agree with idct2" in {
    assert(mnorm(dct20(Mat, true) - dct2(Mat, true)) < tol)
  }

  "dct2j0" should "agree with dct2" in {
    assert(mnorm(dct2j0(Mat) - dct2(Mat)) < tol)
  }

  "idct2j0" should "agree with idct2" in {
    assert(mnorm(dct2j0(Mat, true) - dct2(Mat, true)) < tol)
  }

  "dct2j0" should "invert" in {
     assert(mnorm(dct2j0(dct2j0(Mat), true) - Mat) < tol)
  }

  "dct2j" should "agree with dct2" in {
    assert(mnorm(dct2j(Mat) - dct2(Mat)) < tol)
  }






// eof

