
name := "Filter"

version := "0.1"

scalaVersion := "2.11.7"

libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.2.1"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0" % "provided"

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)







