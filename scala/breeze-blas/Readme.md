# Using optimised BLAS and LAPACK libraries with Scala Breeze

[Breeze](https://github.com/scalanlp/breeze) is the standard scientific and numerical library for [Scala](https://www.scala-lang.org/). For linear algebra operations, it builds on top of the Java library, [netlib](https://github.com/luhenry/netlib). This provides a nice interface to BLAS and related libraries which allows the use of native optimised libraries and will also gracefully fall back to using pure Java implementations if optimised native code libraries can't be found. This is great, since it leads to good code portability, but the Java implementations will typically be slower than optimised native libraries for large matrices, so if you care about speed it is important to install optimised libraries on your system and configure netlib to use them.

See the [netlib Readme](https://github.com/luhenry/netlib/blob/master/README.md) for details of installing native libraries and setting the relevant system properties. In many cases netlib will automatically detect and use native libraries, but it's not foolproof, especially if you are using nonstandard libraries, or if you have multiple libraries installed and you want to specify which one to use. Briefly, you can override default settings by setting the properties `blas`, `lapack` and `arpack`. Each of this can be set by using either `nativeLib`, to specify the name of a library in your system library search path or `nativeLibPath`, to set the full path to the library you require. Full examples of the two approaches are:
```
-Ddev.ludovic.netlib.blas.nativeLib=libopenblas.so

-Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib/x86_64-linux-gnu/libopenblas.so
```
Obviously these need to be customised to your requirements. `lapack` and `arpack` properties are set similarly.

What the netlib readme doesn't discuss is how to set these properties in Scala projects, or how to check/verify the libraries being used in Scala Breeze projects. These are discussed below.

## Setting netlib properties in Scala projects

Exactly how you set system properties depends on exactly how you are building and running your Scala code. However, some build tools will read the environment variable `JAVA_OPTS`, so setting this will very often work. For example, if you are using a bash-like shell, you could set a relevant property with something like:
```bash
export JAVA_OPTS="-Ddev.ludovic.netlib.blas.nativeLib=libblas.so"
```
Multiple properties should be separated with a space.
```bash
export JAVA_OPTS="-Ddev.ludovic.netlib.blas.nativeLib=libblas.so -Ddev.ludovic.netlib.lapack.nativeLib=liblapack.so"
```
Then this might be picked up and used by your build tool. If so, this is often the preferred approach.

### scala-cli

[scala-cli](https://scala-cli.virtuslab.org/) is a popular tool for compiling and running small Scala projects. You can pass in a property directly at the _end_ of the `scala-cli` command-line:
```bash
scala-cli run breeze-test.scala '-Ddev.ludovic.netlib.blas.nativeLib=libblas.so' 
```
Multiple properties should be included separately:
```bash
scala-cli run breeze-test.scala '-Ddev.ludovic.netlib.blas.nativeLib=libblas.so' '-Ddev.ludovic.netlib.lapack.nativeLib=liblapack.so'
```
If you prefer, you can include the option in your Scala source code in the `scala-cli` headers, which might then look similar to:
```scala
//> using scala 3.3.0
//> using dep org.scalanlp::breeze:2.1.0
//> using dep org.scalanlp::breeze-viz:2.1.0
//> using javaOpt -Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0
```
The (significant) disadvantage of this approach is that it makes the code less portable.

### sbt

Many larger Scala project are built using [sbt](https://www.scala-sbt.org/). 
`sbt` checks the `JAVA_OPTS` environment variable, so this is often the preferred way to configure the libraries that you want to use.

Alternatively, you can explicitly set them at the `sbt` command line by inserting them _before_ the required task:
```bash
sbt -Ddev.ludovic.netlib.blas.nativeLib=libblas.so run
```
If you prefer, you can include the options inside your `build.sbt` file with something like:
```scala
javaOptions ++= Seq(
  "-Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0"
)

fork := true
```
Note that you will probably need to fork the project, as the options will be applied to the forked process. But in addition to the usual pros and cons of forking `sbt` projects, this approach has the disadvantage of making the code less portable.

## Verifying netlib instances in Scala Breeze projects

Since the linking to specific libraries happens at runtime, it is often desirable to be able to check whether native libraries are being used within a running Scala Breeze application. There are various ways to do this, but the basic idea can be illustrated with the following Scala code snippet.
```scala
import dev.ludovic.netlib.blas.BLAS
print("BLAS: ")
println(BLAS.getInstance().getClass().getName())
import dev.ludovic.netlib.lapack.LAPACK
print("LAPACK: ")
println(LAPACK.getInstance().getClass().getName())
import dev.ludovic.netlib.arpack.ARPACK
print("ARPACK: ")
println(ARPACK.getInstance().getClass().getName())
```
If the output from this snippet is something like:
```
BLAS: dev.ludovic.netlib.blas.JNIBLAS
LAPACK: dev.ludovic.netlib.lapack.JNILAPACK
ARPACK: dev.ludovic.netlib.arpack.JNIARPACK
```
then you are using native libraries. The letters `JNI` in the final word indicate the use of the "Java native interface". Any other output indicates that a relevant native library has not been found, and the precise output will give some indication of exactly what pure Java implementation has been fallen back to.

If you wanted some more friendly output, you can do something like:
```scala
    val blas = BLAS.getInstance().getClass().getName()
    println(s"Using BLAS: $blas")
    blas match
      case "dev.ludovic.netlib.blas.JNIBLAS" => println("This is a native BLAS of some sort")
      case "dev.ludovic.netlib.blas.VectorBLAS" => println("This is the VectorBLAS for Java 16+")
      case _ => println("Fallen back to a Java BLAS of some sort (probably slow)")
```
Note that this also detects the use of `VectorBLAS`, discussed below.

## Java 16+

As explained in the netlib readme, there is an additional wrinkle if you are using a recent JVM (version 16 or higher). Since recent JVMs expose vector operations, it is now possible to write pure Java BLAS libraries with similar performance to native libraries. Recent versions of netlib include such an implementation, called `VectorBLAS`. So, if you run your Scala Breeze application on a recent JVM, netlib will first check to see if it can detect a `VectorBLAS`, and if it finds it, it will use it in preference to a native library. However, if it can't (which is likely to be the case by default), then it will next look for a *native* BLAS, before eventually falling back to a less performant Java BLAS library. So if you have a good native BLAS installed and configured, then you are probably happy with this, and can safely ignore any errors or warnings about not being able to find a `VectorBLAS`.


