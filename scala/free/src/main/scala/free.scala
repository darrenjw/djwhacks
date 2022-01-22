/*
free.scala


*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

object Concrete:

  import cats.data.State
  type IDState[A] = State[Int, A]
  def createID(): IDState[String] = State(counter =>
    (counter+1, "user"+counter))

  val people = List("Andrew", "Betty", "Charles", "Doris")
  val prog = people.traverse(
    p => createID().map(id => (p, id)) )
  val ids = prog.runA(1).value


object UsingFree:

  enum IDState[A]:
    case CreateID() extends IDState[String]
  import IDState.*

  import cats.free.Free
  type IDStateF[A] = Free[IDState,A]

  import cats.free.Free.liftF
  def createID(): IDStateF[String] =
    liftF[IDState, String](CreateID())

  val people = List("Andrew", "Betty", "Charles", "Doris")
  val prog = people.traverse(
    p => createID().map(id => (p, id)) )
  
  // compiler for the State monad
  import cats.data.State
  type IDStateM[A] = State[Int, A]
  val csm = new (IDState ~> IDStateM):
    def apply[A](fa: IDState[A]): IDStateM[A] = fa match
      case CreateID() => 
        State(counter => (counter+1, "user"+counter))

  val compiled = prog.foldMap(csm)
  val ids = compiled.runA(1).value


object TaglessFinal:

  trait IDState[F[_]]:
    def createID(): F[String]

  def prog[F[_]: Monad](ids: IDState[F]) =
    import ids.*  
    val people = List("Andrew", "Betty", "Charles", "Doris")
    people.traverse(
      p => createID().map(id => (p, id)) )
    
  // compiler for the State monad
  import cats.data.State
  type IDStateM[A] = State[Int, A]
  object IDStateC extends IDState[IDStateM]:
    def createID(): IDStateM[String] = State(counter =>
      (counter+1, "user"+counter))

  val compiled = prog(IDStateC)
  val ids = compiled.runA(1).value


object FreeApp extends IOApp.Simple:

  def display(s: String) = IO { println(s) }

  def run = for
    _ <- display("Hello")
    _ <- display(Concrete.prog.toString)
    _ <- display(Concrete.ids.toString)
    _ <- display(UsingFree.prog.toString)
    _ <- display(UsingFree.ids.toString)
    _ <- display(TaglessFinal.ids.toString)
    _ <- display("Goodbye")
  yield ()

