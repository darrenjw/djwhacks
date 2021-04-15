name := "evilplot-examples"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.1.0-SNAP13" % "test",
  "io.github.cibotech" %% "evilplot" % "0.8.1",
  "io.github.cibotech" %% "evilplot-repl" % "0.8.1",
  "org.scalanlp" %% "breeze" % "1.1",
  // "org.scalanlp" %% "breeze-viz" % "1.1",
  "org.scalanlp" %% "breeze-natives" % "1.1"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "2.13.5"

fork := true


