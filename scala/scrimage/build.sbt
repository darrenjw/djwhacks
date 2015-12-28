name := "scrimage-test"

version := "0.1"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

libraryDependencies += "com.sksamuel.scrimage" %% "scrimage-core" % "2.1.0"
libraryDependencies += "com.sksamuel.scrimage" %% "scrimage-io-extra" % "2.1.0"
libraryDependencies += "com.sksamuel.scrimage" %% "scrimage-filters" % "2.1.0"

scalaVersion := "2.11.7"


