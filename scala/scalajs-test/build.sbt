enablePlugins(ScalaJSPlugin)

name := "ScalaJSExampleApp"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies ++= Seq(
  "org.scala-js" %%% "scalajs-dom" % "1.1.0"
)

scalaVersion := "2.13.4"


