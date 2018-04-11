package effects

import cats.effect.IO

object Main extends App {

  val ioa = IO { println("hey!") }

  val program: IO[Unit] =
    for {
       _ <- ioa
       _ <- ioa
    } yield ()

  program.unsafeRunSync()

}
