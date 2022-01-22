/*
free.scala


*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}



object FreeApp extends IOApp.Simple:

  def display(s: String) = IO { println(s) }

  def run = for
    _ <- display("Hello")
    _ <- display("Goodbye")
  yield ()

