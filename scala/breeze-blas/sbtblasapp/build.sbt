name := "sbtblasapp"

version := "0.1-SNAPSHOT"

scalacOptions ++= Seq(
  "-unchecked", "-deprecation", "-feature"
)

//javaOptions ++= Seq(
//  "-Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0"
//)

libraryDependencies  ++= Seq(
  "org.scalatest" %% "scalatest" % "3.2.16" % "test",
  "org.scalanlp" %% "breeze" % "2.1.0",
  "dev.ludovic.netlib" % "blas" % "3.0.3" // Contains VectorBLAS ??
  // "org.scalanlp" %% "breeze-viz" % "2.1.0"
)

resolvers ++= Seq(
  "Sonatype Snapshots" at
    "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at
    "https://oss.sonatype.org/content/repositories/releases/"
)

scalaVersion := "3.3.1"

//fork := true

