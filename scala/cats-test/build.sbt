name := "cats-test"

version := "0.1"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature", "-Ypartial-unification")

libraryDependencies  ++= Seq(
            "org.typelevel" %% "cats-core" % "1.1.0"
)

scalaVersion := "2.12.4"




