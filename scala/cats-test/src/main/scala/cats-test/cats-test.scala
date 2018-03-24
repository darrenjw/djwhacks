object CatsTest {

  import cats._
  import cats.data._
  import cats.syntax._
  import cats.implicits._

  //implicit val listMonad = new Monad[List] {
  //  def flatMap[A, B](fa: List[A])(f: A => List[B]): List[B] = fa.flatMap(f)
  //  def pure[A](a: A): List[A] = List(a)
  //}

  def main(args: Array[String]): Unit = {
    println("My test of cats...")

    def f1(x: Int): List[Int] = List(x,x)
    def f2(x: Int): List[Int] = List(x,x+1)
    val k1 = Kleisli(f1)
    val k2 = Kleisli(f2)
    val l1 = List(1,10)
    println(l1 flatMap f1 flatMap f2)
    println(l1 >>= f1 >>= f2)
    //println(l1 >>= (f1 _ >=> f2 _))
    println(l1 >>= (k2 compose k1).run)
    println(l1 >>= (k2 <<< k1).run)
    println(l1 >>= (k1 andThen k2).run)
    println(l1 >>= (k1 >>> k2).run)
    println()
    println(l1 flatMap f2 flatMap f1)
    println(l1 >>= f2 >>= f1)
    println(l1 >>= (k1 compose k2).run)
    println(l1 >>= (k1 <<< k2).run)
    println(l1 >>= (k2 andThen k1).run)
    println(l1 >>= (k2 >>> k1).run)


    println("Done.")
  }

}

