name := "rainier-test"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "com.stripe" %% "rainier-core" % "0.1.0"
  //"org.scalanlp" %% "breeze" % "0.13",
  // "org.scalanlp" %% "breeze-viz" % "0.13",
  //"org.scalanlp" %% "breeze-natives" % "0.13"
)


scalaVersion := "2.12.4"


