name := "a-star"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature", "-language:higherKinds",
  "-Ymacro-annotations", "-language:implicitConversions"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.1.0-SNAP13" % "test",
 // "org.typelevel" %% "cats-core" % "2.0.0",
  "org.typelevel" %% "simulacrum" % "1.0.0"
)

val circeVersion = "0.12.1"

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

scalaVersion := "2.13.0"

