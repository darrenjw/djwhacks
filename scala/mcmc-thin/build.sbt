// build.sbt

name := "mcmc"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature", "-language:higherKinds",
  "-language:implicitConversions", "-Ykind-projector:underscores"
)

enablePlugins(ScalaNativePlugin)

enablePlugins(MdocPlugin)

libraryDependencies  ++= Seq(
  "com.github.scopt" %%% "scopt" % "4.0.1",
  ("org.typelevel" %%% "cats-core" % "2.7.0").cross(CrossVersion.for3Use2_13)
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "3.1.0"


// eof

