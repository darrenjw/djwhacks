import scalaz.{Free, ~>, Id, Coyoneda}

object Expression {
  sealed trait Expression[A]
  final case class Add(l: Int, r: Int) extends Expression[Int]
  final case class Mult(l: Int, r: Int) extends Expression[Int]
  final case class Append(l: String, r: String) extends Expression[String]
  final case class Pure[A](a: A) extends Expression[A]

  type Expressable[A] = Coyoneda[Expression, A] 
  object Expression {
    def pure[A](a: A): Free[Expressable, A] =
      Free.liftFC(Pure(a) : Expression[A])

    def add(l: Int, r: Int): Free[Expressable, Int] =
      Free.liftFC(Add(l, r) : Expression[Int])

    def multiply(l: Int, r: Int): Free[Expressable, Int] =
      Free.liftFC(Mult(l, r) : Expression[Int])

    def append(l: String, r: String): Free[Expressable, String] =
      Free.liftFC(Append(l, r) : Expression[String])
  }

  object IdInterpreter extends (Expression ~> Id.Id) {
    def apply[A](exp: Expression[A]): Id.Id[A] =
      exp match {
        case Pure(i) => i
        case Add(l, r)  => l + r
        case Mult(l, r) => l * r
        case Append(l, r) => l ++ r
      }
  }

  import scalaz.std.string._
  import scalaz.std.list._
  import scalaz.Writer
  import scalaz.syntax.writer._
  type Log[A] = Writer[List[String],A]
  object DebugInterpreter extends (Expression ~> Log) {
    def apply[A](exp: Expression[A]): Log[A] =
      exp match {
        case Pure(i)      => List(s"Pure $i").tell.map(_ => i)
        case Add(l, r)    => List(s"Add $l $r").tell.map(_ => l + r)
        case Mult(l, r)   => List(s"Mult $l $r").tell.map(_ => l * r)
        case Append(l, r) => List(s"Append $l $r").tell.map(_ => l ++ r)
      }
  }

  import scalaz.State
  type Counter[A] = State[Int,A]
  object CountingInterpreter extends (Expression ~> Counter) {
    def apply[A](exp: Expression[A]): Counter[A] =
      exp match {
        case Pure(i)      => State.modify((i: Int) => i + 1).map(_ => i)
        case Add(l, r)    => State.modify((i: Int) => i + 1).map(_ => l + r)
        case Mult(l, r)   => State.modify((i: Int) => i + 1).map(_ => l * r)
        case Append(l, r) => State.modify((i: Int) => i + 1).map(_ => l ++ r)
      }
  }


  object Example {
    def calculation =
      //Expression.times(Literal(1), Literal(2))
      for {
        a <- Expression.pure(1)
        b <- Expression.pure(2)
        c <- Expression.multiply(a, b)
        d <- Expression.append(c.toString, b.toString)
      } yield d

    def runId =
      Free.runFC(calculation)(IdInterpreter)

    def runLog =
      Free.runFC(calculation)(DebugInterpreter)

    def runCount =
      Free.runFC(calculation)(CountingInterpreter).run(0)
  }
}
