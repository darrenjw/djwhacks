name := "evilplot-test"

version := "0.1"

scalacOptions ++= Seq("-unchecked", "-deprecation", "-feature")

libraryDependencies  ++= Seq(
	"com.cibo" %% "evilplot" % "0.2.0",
	"com.github.darrenjw" %% "scala-view" % "0.6-SNAPSHOT"
)

resolvers ++= Seq(
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
            "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
	Resolver.bintrayRepo("cibotech", "public")
)

scalaVersion := "2.12.8"

fork := true



