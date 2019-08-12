name := "min-ppl"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)


libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.1.0-SNAP13" % "test",
  "org.scalanlp" %% "breeze" % "1.0",
  "org.scalanlp" %% "breeze-viz" % "1.0",
  "org.scalanlp" %% "breeze-natives" % "1.0",
  "org.typelevel" %% "cats-core" % "2.0.0-M4"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "2.13.0"

scalaVersion in ThisBuild := "2.13.0"
