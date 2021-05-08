val scala3Version = "3.0.0-RC3"

lazy val root = project
  .in(file("."))
  .settings(
    name := "scala3-simple",
    version := "0.1.0",
    scalaVersion := scala3Version,
    libraryDependencies += "org.scalanlp" %% "breeze" % "2.0-SNAPSHOT",
    libraryDependencies += "com.novocode" % "junit-interface" % "0.11" % "test",
    resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
  )
