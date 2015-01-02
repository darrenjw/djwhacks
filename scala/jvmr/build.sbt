name := "gibbs"

version := "0.1"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

libraryDependencies  ++= Seq(
            "org.scalanlp" %% "breeze" % "0.10",
            "org.scalanlp" %% "breeze-natives" % "0.10"
)

resolvers ++= Seq(
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "2.11.2"


// printClasspath task required for sbtInit() in R

lazy val printClasspath = taskKey[Unit]("Dump classpath")

printClasspath := {
  (fullClasspath in Runtime value) foreach {
    e => print(e.data+"!")
  }
}

// end of printClasspath task definition



