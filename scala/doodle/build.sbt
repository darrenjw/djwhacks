name := "doodle-test"

version := "0.1"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

libraryDependencies  ++= Seq(
            "org.scalacheck" %% "scalacheck" % "1.11.4" % "test",
            "org.scalatest" %% "scalatest" % "2.1.7" % "test",
            "underscoreio" %% "doodle" % "0.2.0-31f359-snapshot" 
)

resolvers ++= Seq(
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
            "Underscore Training" at "https://dl.bintray.com/underscoreio/training"
)

scalaVersion := "2.11.6"

initialCommands in console := """
  |import doodle.core._
  |import doodle.syntax._
  |import doodle.jvm._
  |import doodle.examples._
  |import doodle.jvm.Java2DCanvas._
  |import doodle.backend.StandardInterpreter._
""".trim.stripMargin




