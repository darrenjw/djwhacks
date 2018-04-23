import org.scalacheck.Properties
import org.scalacheck.Prop.forAll

object StringSpecification extends Properties("String") {

  property("startsWith") = forAll { (a: String, b: String) =>
    (a+b).startsWith(a)
  }

  property("concatenate") = forAll { (a: String, b: String) =>
    (a+b).length >= a.length && (a+b).length >= b.length
  }

  property("substring") = forAll { (a: String, b: String, c: String) =>
    (a+b+c).substring(a.length, a.length+b.length) == b
  }

}

object SqrtSpec extends Properties("sqrt") {

  import org.scalacheck.Gen

  val smallInteger = Gen.choose(0,10000)

  property("sqrt1") = forAll(smallInteger) { n =>
    math.sqrt(n*n) == n
  }

  import org.scalacheck.Prop.BooleanOperators

  property("sqrt2") = forAll { n: Int =>
    (n >= 0 && n < 10000) ==> (math.sqrt(n*n) == n)
  }

  val tol = 1e-8

  property("sqrt3") = forAll { x: Double =>
    (x >= 0 && x < 10000) ==> (math.abs(math.sqrt(x*x) - x) < tol)
  }

  property("sqrt4") = forAll { x: Double =>
    (x >= 0 && x < 1e7) ==> {
      val sx = math.sqrt(x)
      math.abs(sx*sx - x) < tol
    }
  }

}
