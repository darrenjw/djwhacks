/*
lamp-test.scala
Test of lamp
*/

import cats.*
import cats.implicits.*
import cats.effect.{IO, IOApp}

import lamp.*

object LampApp extends IOApp.Simple:

  def run = IO{

    println("lamp test")
    val tCpu = aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat)
    println(tCpu)
    val tGpu = aten.ATen.eye_0(2L,aten.TensorOptions.dtypeFloat.cuda)
    println(tGpu)

    Scope.root{ implicit scope =>

      // a 3D tensor, e.g. a color image
      val img : STen = STen.rand(List(768, 1024, 3))
      println(img)
      // get its shape
      assert(img.shape == List(768, 1024, 3))

      // select a channel
      assert(img.select(dim=2,index=0).shape == List(768, 1024))

      // manipulate with a broadcasting operation
      val img2 = img / 2d
      println(img2)
      // take max, returns a tensor with 0 dimensions i.e. a scalar
      assert(img2.max.shape == Nil)

      // get a handle to metadata about data type, data layout and device
      assert(img.options.isCPU)

      val vec = STen.fromDoubleArray(Array(2d,1d,3d),List(3),CPU,DoublePrecision)
      println(vec)

      // broadcasting matrix multiplication
      val singleChannel = (img matmul vec)
      assert(singleChannel.shape == List(768L,1024L))

      // svd
      val (u,s,vt) = singleChannel.svd(false)
      assert(u.shape == List(768,768))
      assert(s.shape == List(768))
      assert(vt.shape == List(768,1024))

      val errorOfSVD = (singleChannel - ((u * s) matmul vt)).norm2(dim=List(0,1), keepDim=false)
      assert(errorOfSVD.toDoubleArray.head < 1E-6)
    }      




  }

