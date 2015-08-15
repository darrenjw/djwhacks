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
 
scalaVersion := "2.11.6"


