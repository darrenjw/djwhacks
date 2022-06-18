name := "mandel-js"

enablePlugins(ScalaJSPlugin)

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies ++= Seq(
  "org.scalameta" %%% "munit" % "0.7.29" % Test,
  "org.scala-js" %%% "scalajs-dom" % "2.1.0",
  "org.typelevel" %%% "cats-effect" % "3.3.12",
  "org.typelevel" %%% "cats-core" % "2.8.0",
  "org.typelevel" %%% "spire" % "0.18.0",
  "dev.optics" %%% "monocle-core"  % "3.1.0",
  "dev.optics" %%% "monocle-macro"  % "3.1.0"
)

scalaVersion := "3.1.0"


