// build.sbt

name := "milan-test"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature", "-language:higherKinds",
  "-language:implicitConversions", "-Ypartial-unification"
)

addCompilerPlugin("org.typelevel" %% "kind-projector" % "0.11.0" cross CrossVersion.full)
addCompilerPlugin("org.scalamacros" %% "paradise" % "2.1.1" cross CrossVersion.full)

libraryDependencies  ++= Seq(
//  "org.scalatest" %% "scalatest" % "3.0.8" % "test",
//  "org.scalactic" %% "scalactic" % "3.0.8",
  "org.typelevel" %% "cats-core" % "2.0.0",
  "com.amazon.milan" % "milan-flink" % "0.8-SNAPSHOT", // need in local maven cache
  "com.amazon.milan" % "milan-lang" % "0.8-SNAPSHOT",
  "com.amazon.milan" % "milan-typeutil" % "0.8-SNAPSHOT",
//  "org.apache.flink" % "flink-java" % "1.7.1",
//  "org.apache.flink" % "flink-java" % "1.7.1",
//  "org.apache.flink" % "flink-core" % "1.7.1",
//  "org.apache.flink" %% "flink-scala" % "1.7.1",
//  "org.apache.flink" % "flink-connector-kinesis_2.11" % "1.7-SNAPSHOT",
  "org.slf4j" % "slf4j-api" % "1.7.25",
//  "org.slf4j" % "slf4j-simple" % "1.7.25",
//  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.2",
//  "org.apache.logging.log4j" % "log4j-slf4j-impl" % "2.11.1",
//  "org.apache.logging.log4j" % "log4j-core" % "2.11.1"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

resolvers += Resolver.mavenLocal // milan and flink in my local maven cache

scalaVersion := "2.12.10"



// eof

