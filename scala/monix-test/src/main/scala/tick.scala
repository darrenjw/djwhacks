/*
tick.scala

Simple tick example from the monix tutorial

 */


object TickApp {

  def main(args: Array[String]): Unit = {

    // We need a Scheduler in scope in order to make
    // the Observable produce elements when subscribed
    import monix.execution.Scheduler.Implicits.global
    import monix.reactive._

    import concurrent.duration._

    // We first build an observable that emits a tick per second,
    // the series of elements being an auto-incremented long
    val source = Observable.interval(1.second)
    // Filtering out odd numbers, making it emit every 2 seconds
      .filter(_ % 2 == 0)
    // We then make it emit the same element twice
      .flatMap(x => Observable(x, x))
    // This stream would be infinite, so we limit it to 10 items
      .take(10)

    // Observables are lazy, nothing happens until you subscribe...
    val cancelable = source
    // On consuming it, we want to dump the contents to stdout
    // for debugging purposes
      .dump("O")
    // Finally, start consuming it
      .subscribe()

  }

}


// eof

