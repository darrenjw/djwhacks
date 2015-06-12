
name := "SimpleSparkJoin"

version := "0.1"

scalaVersion := "2.10.4"

libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.2.1"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.0" % "provided"

assemblyJarName in assembly := "simple-join-assembly.jar"

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)







