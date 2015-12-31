/*
foldmap.scala

Parallel foldMap a la Map-Reduce using scalaz Task and applicatives

 */

import scalaz.{Foldable, Monoid}
import scalaz.std.string._
import scalaz.std.anyVal._
import scalaz.syntax.monoid._
import scalaz.concurrent.{Strategy, Task}
import java.util.concurrent.ExecutorService

object FoldMap {

  def foldMapP[A, B: Monoid](seq: Iterable[A])(f: A => B = (a: A) => a)(implicit e: ExecutorService = Strategy.DefaultExecutorService): Task[B] = {
    // Step 1. Break into chunks of works
    val nCpus = Runtime.getRuntime.availableProcessors
    val size = seq.size
    val chunkSize =
      (size.toDouble / nCpus.toDouble).ceil.toInt
    val chunks: Iterator[Iterable[A]] =
      seq.grouped(chunkSize)

    // Step 2. Do each chunk in parallel
    val tasks: Iterator[Task[B]] =
      chunks.map(chunk => Task(foldMap(chunk)(f)))

    // Step 3. Combine the results of parallel work
    // Parallel composition of "Futures"
    // val task: Task[List[B]] =
    //   Task.gatherUnordered(tasks.toSeq)
    // val result: Task[B] = task.map(foldMap(_)())

    // import scalaz.syntax.applicative._
    // tasks.foldLeft(Task.now(mzero[B])){ (accum, task) =>
    //   (accum |@| task){ (a, b) => a |+| b }
    // }

    // Sequence: G[F[A]] => F[G[A]]
    // List[Task[B]] => Task[List[B]]
    // G has a traverse (like a sequence)
    // F has an applicative
    import scalaz.std.list._
    import scalaz.syntax.traverse._
    tasks.toList.sequence.map(foldMap(_)())
  }

  def foldMap[A, B: Monoid](seq: Iterable[A])(f: A => B = (a: A) => a): B =
    seq.foldLeft(mzero[B])(_ |+| f(_))

  def examples() = {
    import java.security.MessageDigest
    val digester = MessageDigest.getInstance("MD5")

    println(foldMap(List(1, 2, 3))() == 6)

    println(foldMap(Seq(1, 2, 3))(_.toString + "! ") == "1! 2! 3! ")
    println(foldMap("Hello world!")(_.toString.toUpperCase) == "HELLO WORLD!")

    val data = for (i <- 1 to 10000) yield (i.toString)
    val work = (s: String) =>
      new String(digester.digest(s.getBytes))

    def time[A](computation: => A): Long = {
      val t0 = System.nanoTime()
      val result = computation
      val t1 = System.nanoTime()
      (t1 - t0)
    }

    println(foldMapP(1 to 1000000)())
    println(time(foldMapP(data)(work).run))
    println(time(foldMap(data)(work)))
  }

  def main(args: Array[String]): Unit = {
    examples()
  }

}

