//> using scala 3.3.0
//> using dep org.scalanlp::breeze:2.1.0
//> using dep org.scalanlp::breeze-viz:2.1.0
//> using javaOpt -Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0

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
    import dev.ludovic.netlib.blas.BLAS
    println(BLAS.getInstance().getClass().getName())



// eof
