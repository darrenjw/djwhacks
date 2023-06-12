//> using scala 3.3.0
//> using dep org.scalanlp::breeze:2.1.0

// cli-test.scala

// run with:
// scala-cli cli-test.scala


import breeze.linalg.*
import breeze.numerics.*

object TestApp:

  @main
  def hello() =
    println("Hello")
    println(DenseMatrix.eye[Double](5))

// eof

