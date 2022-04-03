// build.sbt

name := "sn-example"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature", "-language:higherKinds",
  "-language:implicitConversions", "-Ykind-projector:underscores"
)

enablePlugins(ScalaNativePlugin)

enablePlugins(MdocPlugin)

libraryDependencies  ++= Seq(
  //"org.scalameta" %%% "munit" % "0.7.29" % Test,
  //"org.scalameta" %%% "munit-scalacheck" % "0.7.29" % Test,
  //"org.typelevel" %%% "discipline-munit" % "1.0.9" % Test,
  "com.github.scopt" %%% "scopt" % "4.0.1",
  ("org.typelevel" %%% "cats-core" % "2.7.0").cross(CrossVersion.for3Use2_13)
  //"org.typelevel" %%% "cats-free" % "2.7.0",
  //"org.typelevel" %%% "cats-laws" % "2.7.0",
  //"org.typelevel" %%% "discipline-core" % "1.1.5"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "3.1.0"


// eof

