name := "rainier-test"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "com.stripe" %% "rainier-core" % "0.2.3",
  "com.stripe" %% "rainier-plot" % "0.2.3",
//  "com.cibo" %% "evilplot" % "0.2.0",
  "org.scalanlp" %% "breeze" % "0.13",
  "org.scalanlp" %% "breeze-viz" % "0.13",
  "org.scalanlp" %% "breeze-natives" % "0.13"
)

resolvers ++= Seq(
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

resolvers += Resolver.bintrayRepo("cibotech", "public")

scalaVersion := "2.12.8"

scalaVersion in ThisBuild := "2.12.8" // for ensime

fork := true

