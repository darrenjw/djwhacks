// no enclosing object?!


// scala 2 braces:
def greet(): Unit = {
  println("Hello")
  println("There")
}

// brace free!:
def greet2(): Unit =
  println("Hello")
  println("There2")

// union types
def greet3(n: Int | String): Unit = n match
  case _: Int => println(n.toString)
  case _: String => println(n)

// enums
enum GLM {
  case Binomial, Poisson
}

case class Person(name: String) {
  def hi = s"Hi, $name"
}

// extension method(!)
extension (person: Person)
  def greet(greeting: String): String = s"$greeting, ${person.name}"


def msg = "I was compiled by Scala 3. :)"

@main def hello: Unit = {
  println("Hello world!")
  println(msg)
  greet()
  greet2()
  greet3(42)
  greet3("yay!")
  val fred = Person("Fred")
  println(fred.hi)
  println(fred.greet("Hello"))

}

