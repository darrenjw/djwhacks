name := "cats-state"
version := "0.0.1-SNAPSHOT"

scalaVersion := "2.12.3"

scalacOptions ++= Seq(
  "-encoding", "UTF-8",   // source files are in UTF-8
  "-deprecation",         // warn about use of deprecated APIs
  "-unchecked",           // warn about unchecked type parameters
  "-feature",             // warn about misused language features
  "-language:higherKinds",// allow higher kinded types without `import scala.language.higherKinds`
  "-Xlint",               // enable handy linter warnings
  "-Xfatal-warnings",     // turn compiler warnings into errors
  "-Ypartial-unification" // allow the compiler to unify type constructors of different arities
)

libraryDependencies += "org.typelevel" %% "cats-core" % "1.0.0-MF"

addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.9.3")
