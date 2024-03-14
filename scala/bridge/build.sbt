// build.sbt

name := "bridge"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature", "-language:higherKinds",
  "-language:implicitConversions"
)

enablePlugins(MdocPlugin)

libraryDependencies  ++= Seq(
  "org.scalameta" %% "munit" % "0.7.29" % Test,
  "org.scalameta" %% "munit-scalacheck" % "0.7.29" % Test,
  "org.typelevel" %% "discipline-munit" % "1.0.9" % Test,
  "org.typelevel" %% "cats-core" % "2.9.0",
  "org.typelevel" %% "cats-free" % "2.9.0",
  "org.typelevel" %% "cats-laws" % "2.9.0",
  "org.typelevel" %% "cats-effect" % "3.5.0",
  "org.typelevel" %% "discipline-core" % "1.5.0",
  "org.scalanlp" %% "breeze" % "2.1.0",
  "org.scalanlp" %% "breeze-viz" % "2.1.0",
  "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
)

val monocleVersion = "3.1.0"
libraryDependencies ++= Seq(
  "dev.optics" %%  "monocle-core"  % monocleVersion,
  "dev.optics" %%  "monocle-law"   % monocleVersion % "test"
)

val circeVersion = "0.14.1"
libraryDependencies ++= Seq(
  "io.circe" %% "circe-core",
  "io.circe" %% "circe-generic",
  "io.circe" %% "circe-parser"
).map(_ % circeVersion)


resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "3.3.0"

fork := true

// eof

