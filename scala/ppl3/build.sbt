// build.sbt

name := "ppl3"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature", "-language:higherKinds",
  "-language:implicitConversions", "-Ykind-projector:underscores"
)

enablePlugins(MdocPlugin)

libraryDependencies  ++= Seq(
  "org.scalameta" %% "munit" % "0.7.29" % Test,
  "org.scalameta" %% "munit-scalacheck" % "0.7.29" % Test,
//  "org.typelevel" %% "discipline-munit" % "1.0.9" % Test,
//  "org.typelevel" %% "cats-core" % "2.7.0",
//  "org.typelevel" %% "cats-free" % "2.7.0",
//  "org.typelevel" %% "cats-laws" % "2.7.0",
//  "org.typelevel" %% "cats-effect" % "3.2.2",
//  "org.typelevel" %% "discipline-core" % "1.1.5",
  ("org.typelevel" %% "cats-core" % "2.1.1").cross(CrossVersion.for3Use2_13),
  ("org.typelevel" %% "cats-free" % "2.1.1").cross(CrossVersion.for3Use2_13),
  "org.scalanlp" %% "breeze" % "2.0.1-RC2"
)

/*

val monocleVersion = "3.0.0"
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

 */

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "3.1.1"


// eof
