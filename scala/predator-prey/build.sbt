name := "predator-prey"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "com.github.darrenjw" %% "monte-scala" % "0.1-SNAPSHOT"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/",
  "Personal mvn repo" at "https://www.staff.ncl.ac.uk/d.j.wilkinson/mvn/"
)

scalaVersion := "2.12.4"

