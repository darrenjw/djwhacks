name := "rscala test"

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

unmanagedJars in Compile += file("/home/ndjw1/R/x86_64-pc-linux-gnu-library/3.2/rscala/java/rscala_2.11-1.0.6.jar")

scalaVersion := "2.11.6"


