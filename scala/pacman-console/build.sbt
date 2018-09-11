name := "pacman-console"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "jline" % "jline" % "2.14.2"
)


scalaVersion := "2.12.1"

