name := "plotlyjs"

enablePlugins(ScalaJSPlugin)
enablePlugins (ScalaJSBundlerPlugin, JSDependenciesPlugin)

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies ++= Seq(
  "org.scalameta" %%% "munit" % "0.7.29" % Test,
  "org.scala-js" %%% "scalajs-dom" % "2.1.0",
  "org.typelevel" %%% "cats-core" % "2.7.0",
  "org.typelevel" %%% "spire" % "0.18.0-M3",
  "dev.optics" %%% "monocle-core"  % "3.1.0",
  "dev.optics" %%% "monocle-macro"  % "3.1.0",
  "com.raquo" %%% "laminar" % "0.14.2",
  "org.openmole" %%% "scala-js-plotlyjs" % "1.6.2"
)

scalaVersion := "3.1.0"


