name := "csv"

version := "0.1"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

libraryDependencies  ++= Seq(
            "org.scalacheck" %% "scalacheck" % "1.10.0" % "test",
            "org.scalatest" %% "scalatest" % "2.0.M5b" % "test",
            "org.scalanlp" % "breeze_2.10" % "0.5.2",
            "org.scalanlp" % "breeze-viz_2.10" % "0.5.2"
)

resolvers ++= Seq(
            // other resolvers here
            // if you want to use snapshot builds (currently 0.6-SNAPSHOT), use this.
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

// Scala 2.9.2 is still supported for 0.2.1, but is dropped afterwards.
// Don't use an earlier version of 2.10, you will probably get weird compiler crashes.
scalaVersion := "2.10.3"




