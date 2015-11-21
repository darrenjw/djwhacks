object CatsTest {

  import cats._

  implicit val listMonad = new Monad[List] {
    def flatMap[A, B](fa: List[A])(f: A => List[B]): List[B] = fa.flatMap(f)
    def pure[A](a: A): List[A] = List(a)
  }

  def main(args: Array[String]): Unit = {
    println("My test of cats..")
  }

}

