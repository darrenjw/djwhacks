name := "regression"

version := "0.1"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

libraryDependencies  ++= Seq(
            "org.scalacheck" %% "scalacheck" % "1.11.4" % "test",
            "org.scalatest" %% "scalatest" % "2.1.7" % "test",
            "org.scalanlp" %% "breeze" % "0.10",
            "org.scalanlp" %% "breeze-natives" % "0.10",
            "org.scalanlp" %% "breeze-viz" % "0.8", 
            "org.scala-saddle" %% "saddle-core" % "1.3.+"
)

resolvers ++= Seq(
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "2.11.1"




