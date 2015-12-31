/*
Ex3-mapreduce.scala

Sequential and parallel folds over an iterable collection over a monoid

Uses scalaz Task

 */

object Exercise3 {

  import scalaz.{Foldable, Monoid}
  import scalaz.concurrent.Task
  import scalaz.syntax.monoid._
  import scalaz.std.string._
  import scalaz.std.anyVal._
  import scalaz.std.option._
  import java.util.concurrent.ExecutorService
  import scalaz.concurrent.Strategy

  object FoldMap {

    def foldMap[A, B: Monoid](seq: Iterable[A])(f: A => B = (a: A) => a): B = {
      seq.foldLeft(mzero[B])(_ |+| f(_))
    }

    //def myFoldMap[F[_]:Foldable,A,B:Monoid](seq: F[A])(f: A => B = (a:A)=>a): B = {
    //  seq.foldMap(f)
    //}

    def foldMapP[A, B: Monoid](seq: Iterable[A])(f: A => B = (a: A) => a)(implicit e: ExecutorService = Strategy.DefaultExecutorService): B = {
      val nCpus = Runtime.getRuntime.availableProcessors
      val size = seq.size
      val chunkSize = ((size.toDouble) / (nCpus.toDouble)).ceil.toInt
      val chunks: Iterator[Iterable[A]] = seq.grouped(chunkSize)
      val tasks = chunks map { chunk => Task(foldMap(chunk)(f(_))) }
      // option 1
      //val task = tasks.gatherUnordered(tasks.toSeq)
      //task.map(foldMap(_)(f)) 

      // option 2
      //tasks.foldLeft(Task.now(mzero[B])){(accum,task) => 
      //  (accum |@| task) { (a,b) => a |@| b } }

      // Sequence; G[F[A]] => F[G[A]]
      // List[Task[B]] => Task[List]]
      // G has a traverse
      // F has an applicative

      // option 3
      import scalaz.std.list._
      import scalaz.syntax.traverse._
      tasks.toList.sequence.map(foldMap(_)()).run
    }

    def examples() = {
      println(foldMap(List(1, 2, 3))())
      println(foldMap(List(1, 2, 3))(_.toString + " "))
      println(foldMapP(1 to 1000000)())
    }

  }

  def main(args: Array[String]): Unit = {
    //val r = (Success(1): Result[Int]).map{_*2}
    //println(r)
    FoldMap.examples()
  }

}
