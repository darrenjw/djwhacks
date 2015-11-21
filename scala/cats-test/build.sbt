name := "cats-test"

version := "0.1"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

libraryDependencies  ++= Seq(
            "org.scalacheck" %% "scalacheck" % "1.11.4" % "test",
            "org.scalatest" %% "scalatest" % "2.1.7" % "test",
            "org.spire-math" %% "cats" % "0.3.0"
)

scalaVersion := "2.11.7"




