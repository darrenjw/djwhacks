// blas detector

import breeze.linalg.*
import dev.ludovic.netlib.blas.BLAS

object Main:

  @main
  def run() = 
    println(BLAS.getInstance().getClass().getName())
    println(DenseMatrix((1.0,0.0),(0.2,2.0)) * DenseVector(3.0, 4.0))
    println(BLAS.getInstance().getClass().getName())
    println("\n\n\n")
    BLAS.getInstance().getClass().getName() match
      case "dev.ludovic.netlib.blas.JNIBLAS" => println("Using a native BLAS of some sort") 
      case _ => println("Fallen back to Java BLAS")
    println("\n\n\n")


// eof
