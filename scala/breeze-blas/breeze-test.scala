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


object Main:

  @main
  def run() = 
    println("\n\n\n")
    import dev.ludovic.netlib.blas.BLAS
    print("BLAS: ")
    println(BLAS.getInstance().getClass().getName())
    import dev.ludovic.netlib.lapack.LAPACK
    print("LAPACK: ")
    println(LAPACK.getInstance().getClass().getName())
    import dev.ludovic.netlib.arpack.ARPACK
    print("ARPACK: ")
    println(ARPACK.getInstance().getClass().getName())
    println("\n\n\n")

    import breeze.linalg.*
    val m = DenseMatrix((1.0,0.0),(0.2,2.0))
    println(m * DenseVector(3.0, 4.0))
    println(svd(m))

    println("\n\n\n")
    val blas = BLAS.getInstance().getClass().getName()
    println(s"Using BLAS: $blas")
    blas match
      case "dev.ludovic.netlib.blas.JNIBLAS" => println("This is a native BLAS of some sort")
      case "dev.ludovic.netlib.blas.VectorBLAS" => println("This is the VectorBLAS for Java 16+")
      case _ => println("Fallen back to a Java BLAS of some sort (probably slow)")
    println("\n\n\n")


// eof
