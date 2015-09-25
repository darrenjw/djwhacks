// Types of DI:
//  - Java-land: Spring, Guice
//  - Constructors
//  - Implicits
//  - Cake pattern <-- don't use it!!!
//  - Reader monad

// Constructor DI looks like this:
// final case class Foo(dependency1: Int, dependency2: Int) {
//   def foo(x: Int): Int = x + 1
// }
//
// def bar(foo: Foo) = foo.foo(2)

// The Reader monad allows us to build a sequence of computations,
// and inject a dependency at the end. Conceptually like this:
//
// val foo1 = (dependency) => (x) => x + 1
// val foo2 = (x) => (dependency) => x + 1
//
// foo2(3) flatMap {result => ???} // (dependency) => ???

object ReaderExample {
  import scalaz.Reader

  val reader1: Reader[String, String] =
    Reader[String,String](name => s"Hello $name").
      map(s => s + " How are you?")

  val reader2: Reader[String, String] =
    reader1.
      flatMap(greeting => Reader[String,String] { name =>
        s"$greeting Your name begins with ${name.head}"
      })

  val reader3: Reader[Int, Int] = for {
    a <- Reader[Int, Int](injected => injected)
  } yield a + 1

  println(reader1.run("Dave"))
  println(reader2.run("Noel"))
  println(reader3.run(1000))
}
