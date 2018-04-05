name := "myScalaFXTest"

organization := "myscalafx"

version := "0.1-SNAPSHOT"

scalaVersion := "2.12.1"

scalacOptions += "-Ypartial-unification"

libraryDependencies ++= Seq(
"org.typelevel" %% "cats-core" % "1.1.0",
  "org.scalafx"   %% "scalafx"   % "8.0.102-R11",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test" //http://www.scalatest.org/download
)



shellPrompt := { state => System.getProperty("user.name") + "> " }

// set the main class for the main 'run' task
// change Compile to Test to set it for 'test:run'
//mainClass in (Compile, run) := Some("my.scalafx.ScalaFXHelloWorld")

// Fork a new JVM for 'run' and 'test:run' to avoid JavaFX double initialization problems
fork := true
