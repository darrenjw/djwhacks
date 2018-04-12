/*
Main.scala

Messing around with the State monad in Cats

*/


package state

import cats.data.State

final case class Seed(long: Long) {
  def next = Seed(long * 6364136223846793005L + 1442695040888963407L)
}

object Main extends App {

  val nextLong: State[Seed,Long] = State((s: Seed) => (s.next,s.long))
  val nextBoolean = nextLong map (l => l >= 0L)
  val nextDie = nextLong map (l => (l.abs % 6).toInt + 1)

  val roll2Dice = for {
    roll1 <- nextDie
    roll2 <- nextDie
  } yield (roll1, roll2)

  println("hello")
  val s = Seed(2L)
  println(s)
  println(s.next)
  println(nextLong.run(s).value)
  println(nextBoolean.run(s).value)
  println(nextBoolean.runA(s).value)
  println(nextDie.runA(s).value)
  println(roll2Dice.runA(s).value)
  println("bye")
}


// eof

