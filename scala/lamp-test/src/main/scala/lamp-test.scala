/*
lamp-test.scala
Test of lamp
*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

object LampApp extends IOApp.Simple:

  def run = IO{

    println("lamp test")
    val tCpu = aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat)
    println(tCpu)
    val tGpu = aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat.cuda)
    println(tGpu)

  }

