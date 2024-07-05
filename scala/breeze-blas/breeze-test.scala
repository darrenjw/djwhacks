//> using scala 3.3.0
//> using dep org.scalanlp::breeze:2.1.0
//> using dep org.scalanlp::breeze-viz:2.1.0


// > using javaOpt -Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0
// > using javaOpt -Ddev.ludovic.netlib.blas.nativeLib=libblas.so


// run with:
// scala-cli breeze-test.scala

// get a repl with:
// scala-cli --dep org.scalanlp::breeze:2.1.0


// https://docs.cloudera.com/best-practices/latest/accelerating-spark-ml-applications/topics/bp-spark-ml-native-math-libraries.html
// https://github.com/luhenry/netlib
// https://github.com/fommil/netlib-java


// -Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0
// -Ddev.ludovic.netlib.blas.nativeLib=libblas.so


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
      case "dev.ludovic.netlib.blas.Java8BLAS" => println("Fallen back to Java BLAS")
      case "dev.ludovic.netlib.blas.JNIBLAS" => println("Using a native BLAS of some sort")
    println("\n\n\n")


// eof
