name := "evilplot-examples"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.1.0-SNAP13" % "test",
  "com.cibo" %% "evilplot" % "0.8.0",
  "com.cibo" %% "evilplot-repl" % "0.8.0",
  "org.scalanlp" %% "breeze" % "1.1",
  // "org.scalanlp" %% "breeze-viz" % "1.1",
  "org.scalanlp" %% "breeze-natives" % "1.1"
)


resolvers += Resolver.bintrayRepo("cibotech", "public")
// https://bintray.com/cibotech/public/evilplot

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "2.12.10"

fork := true


