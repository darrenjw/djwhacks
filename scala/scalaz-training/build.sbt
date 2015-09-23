val scalaz = "org.scalaz" %% "scalaz-core" % "7.1.0"

val scalazConcurrent = "org.scalaz" %% "scalaz-concurrent" % "7.1.0"

scalaVersion := "2.11.3"

scalacOptions ++= Seq("-feature")

libraryDependencies ++= Seq(scalaz, scalazConcurrent)


