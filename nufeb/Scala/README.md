Darren's Scala Code
===================

Note that this repo contains everything that is needed to build and run the Scala code examples on any system that has Java installed. Any recent version of Java is fine. You do not need to "install" Scala or any Scala "packages" in order to run the code. If you have Java and a decent internet connection, you are good to go. This is one of the benefits of Scala - you can run it anywhere, on any system with a Java installation.

To check if you have Java installed, just run:

java -version

at your system command prompt. If you get an error, Java is absent or incorrectly installed. Installing Java is very easy on any platform, but the best way to install it depends on exactly what OS you are running, so search the internet for advice on the best way to install Java on your OS.

The examples use "sbt" - the "simple build tool" for Scala. You can "install" sbt on your system by copying the launcher script and the launch jar to the same directory somewhere on your path. However, even if you don't want to do this or don't know how to do this, you can still run the examples.

For example, from the "shoving" directory, you should be able to type "../sbt.sh" on Linux or any other Unix-like system (including Macs) in order to run sbt. Similarly, on Windows, you should be able to type "..\sbt" in order to run sbt.

From the sbt prompt, typing "run" will compile and run the code, typing "test" will compile and run any tests, typing "doc" will generate the ScalaDoc documentation (which will be stored in ./target/scala-2.xx/api/), and typing "console" will give a Scala REPL with a properly configured classpath for interactive use.



